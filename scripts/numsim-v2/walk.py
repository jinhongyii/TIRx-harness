"""Reflective TVM-FFI tree helpers used by inventory.py."""

import tvm_ffi
from tvm_ffi.container import Array, Map


def fields_of(obj):
    ti = getattr(type(obj), "__tvm_ffi_type_info__", None)
    out = []
    chain = []
    while ti is not None:
        chain.append(ti)
        ti = ti.parent_type_info
    for t in reversed(chain):
        for f in t.fields:
            out.append(f.name)
    return out


def type_key(o):
    ti = getattr(type(o), "__tvm_ffi_type_info__", None)
    return ti.type_key if ti else type(o).__name__


def dump(o, d=0, name="", seen=None, maxd=40):
    seen = set() if seen is None else seen
    ind = "  " * d
    if isinstance(o, (Array, list, tuple)):
        print(f"{ind}{name}: [{len(o)}]")
        for i, x in enumerate(o):
            dump(x, d + 1, str(i), seen, maxd)
        return
    if isinstance(o, Map):
        print(f"{ind}{name}: {{}}")
        for k, v in o.items():
            dump(v, d + 1, str(k), seen, maxd)
        return
    if not isinstance(o, tvm_ffi.Object) or d > maxd:
        print(f"{ind}{name}: {o!r}"[:160])
        return
    tk = type_key(o)
    if tk in ("ir.Span", "ir.SourceName"):
        return
    extra = ""
    if tk == "tirx.Var" or tk == "tirx.SizeVar":
        extra = f" {o.name} {o.dtype}"
    if tk == "tirx.IntImm" or tk == "tirx.FloatImm":
        extra = f" {o.value} {o.dtype}"
    if tk == "ir.Op":
        extra = " " + o.name
    if tk == "tirx.Buffer":
        extra = f" {o.name}"
    print(f"{ind}{name}: {tk}{extra}")
    if id(o) in seen and tk in ("tirx.Var", "tirx.Buffer", "tirx.SizeVar"):
        return
    seen.add(id(o))
    if tk in ("tirx.Var", "tirx.SizeVar", "tirx.IntImm", "tirx.FloatImm", "ir.Op"):
        return
    for fn in fields_of(o):
        if fn == "span":
            continue
        try:
            v = getattr(o, fn)
        except Exception as e:
            v = f"<err {e}>"
        dump(v, d + 1, fn, seen, maxd)
