"""IR inventory over a capture directory (see capture_plugin.py).

Usage: $PY inventory.py CAPTURE_DIR OUT.json   (FILTER=<nodeid regex> to restrict)
Counts node kinds, ops, dtypes, layouts, control-flow shapes per corpus.
"""

import collections
import glob
import json
import os
import sys

import tvm_ffi
from tvm_ffi.container import Array, Map

import tvm

sys.path.insert(0, os.path.dirname(__file__))
from walk import fields_of, type_key

SCOPE_BIND = {
    0: "kernel>cluster",
    1: "kernel>cta",
    2: "cluster>cta",
    3: "cta>warpgroup",
    4: "cta>warp",
    5: "warpgroup>warp",
    6: "warp>thread",
    7: "cta>thread",
    8: "warpgroup>thread",
    9: "cluster>cta_pair",
}
LANE_SCOPES = {6, 7, 8}
WARP_SCOPES = {3, 4, 5}

C = collections.defaultdict(collections.Counter)  # global counters
K = collections.defaultdict(lambda: collections.defaultdict(set))  # feature -> key -> kernels


def note(cat, key, kname, n=1):
    C[cat][key] += n
    K[cat][key].add(kname)


class Walker:
    def __init__(self, kname):
        self.k = kname
        self.lane = set()  # ids of per-lane vars / buffers
        self.warp = set()
        self.seen_bufvars = set()
        self.loop_depth = 0
        self.loopvars = set()

    def vars_in(self, e, acc=None):
        acc = set() if acc is None else acc
        stack = [e]
        while stack:
            o = stack.pop()
            if isinstance(o, (Array, list, tuple)):
                stack.extend(o)
                continue
            if isinstance(o, Map):
                stack.extend(o.values())
                continue
            if not isinstance(o, tvm_ffi.Object):
                continue
            tk = type_key(o)
            if tk in ("ir.Var", "tirx.Var", "ir.SizeVar"):
                acc.add(o)
                continue
            if tk == "ir.Op":
                acc.add(("op", o.name))
                continue
            for fn in fields_of(o):
                if fn in ("span", "ty"):
                    continue
                try:
                    stack.append(getattr(o, fn))
                except Exception:
                    pass
        return acc

    def classify(self, e):
        vs = self.vars_in(e)
        ops = {v[1] for v in vs if isinstance(v, tuple)}
        if any("elect" in o for o in ops):
            return "elect"
        # identity via same_as is unreliable after load; use name+dtype handles
        if any(not isinstance(v, tuple) and self.tag(v) == "lane" for v in vs):
            return "lane-divergent"
        if any(not isinstance(v, tuple) and self.tag(v) == "warp" for v in vs):
            return "warp-varying(cta-divergent)"
        if any("ld" in o or "load" in o or "wait" in o or "shfl" in o or "vote" in o for o in ops):
            return "data-dependent"
        return "uniform"

    def tag(self, v):
        h = v.__hash__()
        if h in self.lane:
            return "lane"
        if h in self.warp:
            return "warp"
        return None

    def bufinfo(self, v):
        ty = getattr(v, "ty", None)
        if ty is None or type_key(ty) != "tirx.BufferType":
            return
        h = v.__hash__()
        if h in self.seen_bufvars:
            return
        self.seen_bufvars.add(h)
        scope = str(ty.storage_scope)
        dt = str(ty.dtype.dtype) if hasattr(ty.dtype, "dtype") else str(ty.dtype)
        note("buffer_scope_dtype", f"{scope}:{dt}", self.k)
        lay = getattr(ty, "layout", None)
        if lay is not None:
            lk = type_key(lay)
            desc = lk
            if lk == "tirx.TileLayout":
                axes = tuple(sorted({str(it.axis.name) for it in lay.shard}))
                desc += f" shard_axes={axes} replica={len(lay.replica)} offset={len(lay.offset)}"
            note("layout", f"{scope}:{desc}", self.k)
            if lk != "tirx.TileLayout":
                note("layout_exotic", f"{scope}:{lk}:{str(lay)[:100]}", self.k)
        note(
            "buffer_shape_dyn",
            f"{scope}:{'static' if all(type_key(x) == 'ir.IntImm' for x in ty.shape) else 'dyn'}",
            self.k,
        )

    def walk(self, o, parent_field=""):
        if isinstance(o, (Array, list, tuple)):
            for x in o:
                self.walk(x, parent_field)
            return
        if isinstance(o, Map):
            for kk, v in o.items():
                self.walk(v, parent_field)
            return
        if not isinstance(o, tvm_ffi.Object):
            return
        tk = type_key(o)
        if tk in ("ir.Span", "ir.SourceName"):
            return
        note("node", tk, self.k)
        ty = getattr(o, "ty", None) if tk not in ("tirx.PrimFunc",) else None
        if tk in ("ir.Var", "tirx.Var", "ir.SizeVar"):
            self.bufinfo(o)
            if ty is not None and type_key(ty) == "ir.PrimType":
                note("var_dtype", str(ty.dtype), self.k)
            return
        if ty is not None and type_key(ty) == "ir.PrimType" and tk != "ir.IntImm":
            note("expr_dtype", str(ty.dtype), self.k)
        if tk == "ir.IntImm" or tk == "ir.FloatImm":
            note("imm_dtype", f"{tk}:{ty.dtype}", self.k)
            return
        if tk == "ir.StringImm" or tk == "tirx.StringImm":
            return
        if tk == "tirx.ScopeIdDefStmt":
            d = getattr(o, "def")
            note("scope_id", SCOPE_BIND.get(int(d.scope), d.scope), self.k)
            for v in d.def_ids:
                if int(d.scope) in LANE_SCOPES:
                    self.lane.add(v.__hash__())
                elif int(d.scope) in WARP_SCOPES:
                    self.warp.add(v.__hash__())
            if d.extents is None:
                note("scope_id_extent", "deferred", self.k)
            else:
                note(
                    "scope_id_extent",
                    "static" if all(type_key(x) == "ir.IntImm" for x in d.extents) else "dynamic",
                    self.k,
                )
        if tk == "tirx.ExecScope":
            note(
                "exec_scope",
                {2: "cluster", 3: "cta", 4: "warpgroup", 5: "warp", 6: "thread"}.get(
                    int(o.kind), o.kind
                ),
                self.k,
            )
        if tk == "tirx.AttrStmt":
            note("attr_key", str(o.attr_key), self.k)
        if tk == "tirx.AllocBuffer" or tk == "tirx.DeclBuffer":
            b = o.buffer
            bty = b.ty
            note(tk.split(".")[1] + "_scope", str(bty.storage_scope), self.k)
            if str(bty.storage_scope) == "local":
                try:
                    n = 1
                    for x in bty.shape:
                        n *= int(x.value)
                    note(
                        "local_size",
                        "1" if n == 1 else "2-8" if n <= 8 else "9-64" if n <= 64 else ">64",
                        self.k,
                    )
                except Exception:
                    note("local_size", "dynamic", self.k)
            for kk in getattr(o, "annotations", None) or {}:
                note("alloc_annotation", str(kk), self.k)
        if tk == "tirx.For":
            ext = o.extent
            stat = "static" if type_key(ext) == "ir.IntImm" else "dynamic"
            note("for_kind", f"{int(o.kind)}:{stat}", self.k)
            for kk in o.annotations if o.annotations is not None else {}:
                note("for_annotation", str(kk), self.k)
            if o.step is not None if hasattr(o, "step") else False:
                note("for_step", "explicit", self.k)
            note("for_extent_class", self.classify(ext) if stat == "dynamic" else "static", self.k)
            if self.loop_depth >= 0:
                note("for_depth", str(self.loop_depth + 1), self.k)
            self.loop_depth += 1
            if int(o.kind) == 3:
                self.loopvars.add(o.loop_var.__hash__())
            for fn in fields_of(o):
                if fn == "span":
                    continue
                self.walk(getattr(o, fn), fn)
            self.loop_depth -= 1
            return
        if tk == "tirx.While":
            note("while_cond", self.classify(o.condition), self.k)
        if tk == "tirx.IfThenElse":
            note("if_cond", self.classify(o.condition), self.k)
            note("if_else", "has_else" if o.else_case is not None else "no_else", self.k)
        if tk == "prim.Select" or tk == "tirx.Select":
            note("select_cond", self.classify(o.condition), self.k)
        if tk == "tirx.Bind" or tk == "tirx.LetStmt" or tk == "tirx.Let":
            v = getattr(o, "var", None)
            if v is not None:
                c = self.classify(o.value)
                if c == "lane-divergent":
                    self.lane.add(v.__hash__())
                elif c.startswith("warp"):
                    self.warp.add(v.__hash__())
        if tk == "tirx.BufferStore":
            # taint local scalar buffers
            c = self.classify(o.value)
            h = o.buffer.__hash__()
            if c == "lane-divergent":
                self.lane.add(h)
            elif c.startswith("warp") and h not in self.lane:
                self.warp.add(h)
            bt = o.buffer.ty
            dtype = bt.dtype.dtype if hasattr(bt.dtype, "dtype") else bt.dtype
            vec = "vec" if any(type_key(i) == "prim.Ramp" for i in o.indices) else "scalar"
            note("store", f"{bt.storage_scope}:{dtype}:{vec}", self.k)
        if tk == "ir.TensorLoad" or tk == "tirx.BufferLoad":
            src = o.source
            if (
                src is not None
                and getattr(src, "ty", None) is not None
                and type_key(src.ty) == "tirx.BufferType"
            ):
                note("load", f"{src.ty.storage_scope}", self.k)
                if str(src.ty.storage_scope) == "local":
                    note(
                        "local_index",
                        "const"
                        if all(type_key(i) == "ir.IntImm" for i in o.indices)
                        else (
                            "loopvar-only"
                            if self.vars_in(o.indices)
                            and all(
                                not isinstance(v, tuple) and v.__hash__() in self.loopvars
                                for v in self.vars_in(o.indices)
                            )
                            else "dynamic"
                        ),
                        self.k,
                    )
        if "op" in fields_of(o):
            op = o.op
            if isinstance(op, tvm_ffi.Object) and type_key(op) == "ir.Op":
                name = str(op.name)
                note("call", f"{tk}|{name}", self.k)
                note("op", name, self.k)
                if tk == "ir.Call" or tk == "tirx.Call":
                    for a in o.args:
                        if type_key(a) in ("ir.StringImm", "tirx.StringImm"):
                            note("call_string_arg", f"{name}:{str(a.value)[:60]}", self.k)
            elif isinstance(op, tvm_ffi.Object):
                note("call", f"{tk}|<{type_key(op)}>", self.k)
        for fn in fields_of(o):
            if fn in ("span", "ty"):
                continue
            try:
                v = getattr(o, fn)
            except Exception:
                continue
            self.walk(v, fn)


def params(f, kname):
    for p in f.params:
        ty = p.ty
        tk = type_key(ty)
        if tk == "tirx.BufferType":
            note("param", f"buffer:{ty.storage_scope}", kname)
        elif tk == "ir.PrimType":
            note("param", f"scalar:{ty.dtype}", kname)
        elif tk == "ir.PointerType":
            note("param", "pointer", kname)
        else:
            note("param", tk, kname)
    if f.attrs:
        for kk in f.attrs.keys():
            note("func_attr", str(kk), kname)


def one(path):
    C.clear()
    K.clear()
    h = os.path.basename(path)[:-5]
    try:
        f = tvm.ir.load_json(open(path).read())
    except Exception as e:
        return (h, None, str(e))
    name = str(f.attrs["global_symbol"]) if f.attrs and "global_symbol" in f.attrs else h
    kname = f"{name}@{h[:6]}"
    params(f, kname)
    w = Walker(kname)
    for p in f.params:
        w.bufinfo(p)
    try:
        w.walk(f.body)
    except Exception as e:
        note("walk_error", repr(e)[:120], kname)
    return (h, kname, {cat: dict(c) for cat, c in C.items()})


def main(capdir, out):
    import multiprocessing as mp

    idx = {}
    for fn in glob.glob(os.path.join(capdir, "index.*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            idx.setdefault(r["hash"], set()).add(r["nodeid"].split("::")[0])
    kernels = []
    import re

    flt = os.environ.get("FILTER")
    paths = sorted(glob.glob(os.path.join(capdir, "*.json")))
    if flt:
        paths = [
            p
            for p in paths
            if any(re.search(flt, n) for n in idx.get(os.path.basename(p)[:-5], ()))
        ]
    with mp.Pool(48) as pool:
        for h, kname, cs in pool.imap_unordered(one, paths, chunksize=4):
            if kname is None:
                print("load fail", h, cs)
                continue
            kernels.append((kname, sorted(idx.get(h, ()))))
            for cat, c in cs.items():
                for k, n in c.items():
                    C[cat][k] += n
                    K[cat][k].add(kname)
    res = {
        "kernels": kernels,
        "counters": {
            cat: {k: [n, len(K[cat][k])] for k, n in c.most_common()} for cat, c in C.items()
        },
        "examples": {cat: {k: sorted(K[cat][k])[:4] for k in c} for cat, c in C.items()},
    }
    json.dump(res, open(out, "w"), indent=1)
    print(len(kernels), "kernels")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
