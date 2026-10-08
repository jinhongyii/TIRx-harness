"""``tirx.ptx.*`` table ops -> ``numsim_core::Instr`` (decision 1).

Every op is decoded with ``ptx_decode`` (TVM's own table), then:

* families with engine-visible semantics map to their dedicated variant
  (``LoadAddr``/``Store``, ``Atom``, ``MbarArrive``, ``Tma``, ``TcgenMma``...);
* the pure-register tail becomes ``Ptx{op, dsts, srcs, pred, keep_dst}`` with
  ``OpKey{name, mods}`` = op name + non-empty modifier tokens in slot order.

Only ``Ptx`` carries a guard predicate; a predicated dedicated op is wrapped
in ``If{pred}``. Shared-space address operands given as generic pointers are
converted with ``Cvta`` (TVM auto-coerces them the same way).

Handlers are looked up by exact table name, then by prefix (``_PREFIX``).
Unknown non-ALU families and contract gaps raise ``_Unsupported``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from . import program_builder as pb
from . import ptx_decode
from .dtypes import type_key
from .memory import _Unsupported

if TYPE_CHECKING:
    from .ir_walk import Lowerer


SPACES = {
    "": "Generic", "global": "Global", "shared": "Shared", "shared::cta": "Shared",
    "shared::cluster": "SharedCluster", "local": "Local", "param": "Param", "param::entry": "Param",
    "const": "Const", "tmem": "Tmem",
}
SEMS = {"": "Weak", "weak": "Weak", "relaxed": "Relaxed", "acquire": "Acquire", "release": "Release",
        "acq_rel": "AcqRel", "sc": "Sc", "volatile": "Volatile", "mmio": "Mmio"}
SCOPES = {"cta": "Cta", "cluster": "Cluster", "gpu": "Gpu", "sys": "Sys"}
ATOM_OPS = {"add": "Add", "min": "Min", "max": "Max", "inc": "Inc", "dec": "Dec", "and": "And", "or": "Or",
            "xor": "Xor", "exch": "Exch", "cas": "Cas"}
REDUX_OPS = {"add": "Add", "min": "Min", "max": "Max", "and": "And", "or": "Or", "xor": "Xor"}
CACHE_OPS = {"": "Default", "ca": "Ca", "cg": "Cg", "cs": "Cs", "lu": "Lu", "cv": "Cv", "wb": "Wb", "wt": "Wt"}
EVICT = {"": "Normal", "L1::evict_normal": "Normal", "L1::evict_first": "First", "L1::evict_last": "Last",
         "L1::evict_unchanged": "Unchanged", "L1::no_allocate": "NoAllocate"}
L2_PREFETCH = {"": 0, "L2::64B": 64, "L2::128B": 128, "L2::256B": 256}
TC_SHAPES = {"32x32b": "S32x32b", "16x64b": "S16x64b", "16x128b": "S16x128b", "16x256b": "S16x256b"}
TCGEN_DESCRIPTOR_ARCHES = frozenset({"sm_100a", "sm_100f", "sm_103a", "sm_103f", "sm_107a", "sm_107f"})
MMA_KINDS = {"kind::f16": "F16", "kind::tf32": "Tf32", "kind::f8f6f4": "F8f6f4", "kind::i8": "I8",
             "kind::mxf8f6f4": "MxF8f6f4", "kind::mxf4": "MxF4", "kind::mxf4nvf4": "MxF4Nvf4",
             "kind::ti16": "Ti16"}
COLLECTOR = {"": "None", "fill": "Fill", "use": "Use", "lastuse": "LastUse", "discard": "Discard"}


class PtxCtx:
    """One decoded call being lowered."""

    def __init__(self, lw: "Lowerer", node: Any, decoded: ptx_decode.DecodedPtx):
        self.lw = lw
        self.node = node
        self.d = decoded
        self.mods = dict(decoded.modifiers)
        self.ops = {info.name: (info, values) for info, values in zip(decoded.operands, decoded.values)}
        self.pred_lanes = {info.name: flags for info, flags in zip(decoded.operands, decoded.pred_lanes)}
        self.write_backs: list[Callable[[], None]] = []
        self.pred = lw.cast_to(lw.expr(decoded.predicate), pb.Ty("Pred")) if decoded.predicate is not None else None
        self._site: int | None = None

    # -- modifiers --------------------------------------------------------
    def mod(self, name: str) -> str:
        return self.mods.get(name, "")

    def flag(self, name: str) -> bool:
        return bool(self.mods.get(name, ""))

    @property
    def name(self) -> str:
        return self.d.table_name

    def sem(self, default: str = "Weak") -> str:
        token = self.mod("sem")
        if self.flag("mmio"):
            return "Mmio"
        return SEMS[token] if token else default

    def scope(self, default: str = "Gpu") -> str:
        token = self.mod("scope")
        return SCOPES[token] if token else default

    def int_mod(self, name: str, prefix: str = "") -> int:
        token = self.mod(name)
        return int(token[len(prefix):]) if token else 0

    # -- operands ---------------------------------------------------------
    def has(self, name: str) -> bool:
        entry = self.ops.get(name)
        if entry is None:
            return False
        info, values = entry
        return info.literal is None and len(values) > 0 and any(v is not ptx_decode.SINK for v in values)

    def info(self, name: str) -> ptx_decode.OperandInfo:
        return self.ops[name][0]

    def nodes(self, name: str) -> tuple[Any, ...]:
        return self.ops[name][1]

    def src(self, name: str) -> pb.Operand:
        values = self.nodes(name)
        if len(values) != 1:
            raise _Unsupported(self.node, f"{self.d.op_name}: operand {name} has {len(values)} lanes")
        return self.read(name, 0, values[0])

    def is_pred_lane(self, name: str, lane: int) -> bool:
        flags = self.pred_lanes.get(name, ())
        return lane < len(flags) and flags[lane]

    def read(self, name: str, lane: int, node: Any) -> pb.Operand:
        """Source lane ``lane`` of operand ``name``. A ``.pred``-class lane is
        bridged as TVM's helper does (``setp.ne.b32 p, %N, 0``): the op sees
        ``carrier != 0`` as 0/1 in the carrier's type, never its raw bits."""
        value = self.lw.expr(node)
        return bool_carrier(self.lw, value) if self.is_pred_lane(name, lane) else value

    def opt_src(self, name: str) -> pb.Operand | None:
        return self.src(name) if self.has(name) else None

    def srcs(self, name: str) -> list[pb.Operand]:
        return [self.read(name, i, v) for i, v in enumerate(self.nodes(name))] if name in self.ops else []

    def uimm(self, name: str, bits: int = 32) -> int:
        value = self.imm(name)
        if not 0 <= value < 1 << bits:
            raise _Unsupported(self.node, f"{self.d.op_name}: immediate {name}={value} out of range")
        return value

    def imm(self, name: str) -> int:
        values = self.nodes(name)
        node = values[0]
        if isinstance(node, str):
            return int(node)
        if type_key(node) != "ir.IntImm":
            raise _Unsupported(self.node, f"{self.d.op_name}: {name} must be a constant")
        return int(node.value)

    def mbar_addr(self, name: str) -> tuple[pb.Operand, str]:
        """An mbarrier operand (W2-13 ruling): with no state-space qualifier the
        PTX form is generic addressing. A 64-bit value stays ``Generic``; a
        32-bit value (e.g. a ``mapa.shared::cluster`` result) is
        ``SharedCluster``. Every shared::cta address is also a valid
        shared::cluster address under rank tagging, so this never forces CTA."""
        info = self.info(name)
        token = info.space if info.space else self.mod("space")
        if token:
            return self.addr(name)
        value = self.src(name)
        if self.lw.operand_ty(value).bits == 64 and not self._cluster_window(value):
            return value, "Generic"
        return self.lw.address_in(value, "SharedCluster"), "SharedCluster"

    def _cluster_window(self, value: pb.Operand) -> bool:
        """Is every writer of ``value`` a ``mapa.shared::cluster`` (a shared::cluster
        window address held in a 64-bit register, not a generic pointer)?"""
        if not isinstance(value, pb.Reg):
            return False
        writers = [i for i in self.lw.builder.code if value in i.writes()]
        return bool(writers) and all(i.variant == "Mapa" and i.space == "SharedCluster" for i in writers)

    def addr(self, name: str, default_space: str | None = None) -> tuple[pb.Operand, str]:
        info = self.info(name)
        token = info.space if info.space else self.mod("space")
        space = SPACES.get(token, "Generic")
        if not token and default_space is not None:
            space = default_space
        value = self.src(name)
        return self.lw.address_in(value, space), space

    def dst(self, name: str) -> pb.Reg | None:
        values = self.nodes(name)
        if len(values) != 1:
            raise _Unsupported(self.node, f"{self.d.op_name}: destination {name} has {len(values)} lanes")
        return self._bind(values[0], self.info(name))

    def dsts(self, name: str) -> list[pb.Reg | None]:
        return [self._bind(v, self.info(name)) for v in self.nodes(name)]

    def _bind(self, node: Any, info: ptx_decode.OperandInfo) -> pb.Reg | None:
        if node is ptx_decode.SINK:
            return None
        ty_hint = pb.Ty.from_tvm(info.dtype) if info.dtype else None
        reg, write_back = self.lw.lvalue_target(node, ty_hint)
        if self.d.table_name.startswith("elect") and type_key(node) == "ir.TensorLoad":
            self.lw.elect_buffers.add(int(node.source.__chandle__()))
        if write_back is not None:
            self.write_backs.append(write_back)
        return reg

    def scratch(self, ty: pb.Ty) -> pb.Reg:
        return self.lw.builder.reg(ty)

    def ptx_ty(self, token_slot: str = "type", lanes: int = 1) -> pb.Ty:
        return pb.Ty.from_ptx(self.mod(token_slot), lanes)

    # -- emission ---------------------------------------------------------
    def site(self) -> int:
        if self._site is None:
            # W5-15: one logical buffer per pointer operand, in PTX operand order
            # (copies: dst then src; then the completion mbarrier).
            pointers = [value for info, values in zip(self.d.operands, self.d.values)
                        if info.kind in ("addr", "ptr") for value in values if value is not ptx_decode.SINK]
            self._site = self.lw.site(self.node, op_name=self.d.op_name, operands=pointers or None)
        return self._site

    def emit(self, variant: str, /, **fields: Any) -> None:
        """Emit one dedicated instruction (predicated via If), then write-backs."""
        b = self.lw.builder
        if_pc = None
        if self.pred is not None:
            if_pc = b.emit("If", site=self.site(), cond=self.pred, else_pc=0, end_pc=0, elect=False)
        b.emit(variant, site=self.site(), **fields)
        self.flush()
        if if_pc is not None:
            end = b.emit("EndIf")
            b.patch(if_pc, "If", cond=self.pred, else_pc=end, end_pc=end, elect=False)
            self.pred = None

    def flush(self) -> None:
        for write_back in self.write_backs:
            write_back()
        self.write_backs = []


def bool_carrier(lw: "Lowerer", value: pb.Operand) -> pb.Operand:
    """``value != 0`` as 0/1 in ``value``'s own type (TVM's ``.pred`` bridge)."""
    ty = lw.operand_ty(value)
    if ty.elem == "Pred":
        return value
    return lw.cast_to(lw.cast_to(value, pb.Ty("Pred")), ty)


def _register_lvalue(lw: "Lowerer", node: Any) -> bool:
    """Is ``node`` a register-array element or a ``local`` buffer element?"""
    from .memory import MemRef, RegArray

    if type_key(node) == "ir.Call" and str(node.op.name) == "tirx.address_of":
        node = node.args[0]
    if type_key(node) != "ir.TensorLoad":
        return False
    ref = lw.ref_of(node.source)
    return isinstance(ref, RegArray) or (isinstance(ref, MemRef) and ref.space == "Local")


def bridge_guarded_pred_destinations(ctx: PtxCtx) -> None:
    """Off-lane value of a guarded op's ``.pred``-class destinations.

    TVM's ``_pred_keep`` helper reads the carrier on every lane
    (``setp.ne.b32 pd, %0, 0``) and writes ``selp.b32 %0, 1, 0, pd`` after the
    guarded instruction, so an off lane ends with ``old != 0``. ``_pred_undef``
    selects from an unwritten predicate: 0 or 1, modeled as 0 without reading
    the carrier. Only register/local destinations are bridged here (the
    carrier is a C lvalue the helper binds by reference); the op itself still
    runs entirely inside the guard."""
    lw = ctx.lw
    for info, values in zip(ctx.d.operands, ctx.d.values):
        if info.rw not in ("w", "rw") or info.literal is not None:
            continue
        for lane, node in enumerate(values):
            if node is ptx_decode.SINK or not ctx.is_pred_lane(info.name, lane):
                continue
            if not _register_lvalue(lw, node):
                continue
            ty_hint = pb.Ty.from_tvm(info.dtype) if info.dtype else None
            reg, write_back = lw.lvalue_target(node, ty_hint)
            ty = lw.operand_ty(reg)
            if ctx.d.preserve_dst:
                value = bool_carrier(lw, lw.expr(node))
            else:
                value = lw.const(ty, 0)
            lw.builder.emit("Mov", dst=reg, src=lw.cast_to(value, ty))
            if write_back is not None:
                write_back()


# ---------------------------------------------------------------------------
# Generic Ptx (pure register tail)
# ---------------------------------------------------------------------------


def lower_generic(c: PtxCtx) -> None:
    lw = c.lw
    dsts: list[pb.Reg] = []
    srcs: list[pb.Operand] = []
    for info, values in zip(c.d.operands, c.d.values):
        if info.literal is not None:
            continue
        for lane, value in enumerate(values):
            if info.kind == "addr":
                # A register op that names an address only uses its value (e.g. createpolicy.range).
                srcs.append(lw.as_address(lw.expr(value)))
                continue
            if info.rw == "r":
                srcs.append(c.read(info.name, lane, value))
                continue
            if value is ptx_decode.SINK:
                dsts.append(c.scratch(pb.Ty.from_tvm(info.dtype) if info.dtype else pb.Ty("U32")))
                continue
            reg = c._bind(value, info)
            if info.rw == "rw":
                srcs.append(reg)
            dsts.append(reg)
    op = lw.builder.op(pb.OpKey(c.d.op_name, c.d.mod_tokens))
    # A guarded op leaves its destinations untouched where the guard is off
    # (legacy `keep` semantics; e.g. a CLC response sentinel survives).
    lw.builder.emit("Ptx", site=c.site(), op=op, dsts=dsts, srcs=srcs, pred=c.pred,
                    keep_dst=c.d.preserve_dst or c.pred is not None)
    if c.pred is not None and c.write_backs:
        # Memory destinations are written back only where the guard held.
        b = lw.builder
        if_pc = b.emit("If", site=c.site(), cond=c.pred, else_pc=0, end_pc=0, elect=False)
        c.flush()
        end = b.emit("EndIf")
        b.patch(if_pc, "If", cond=c.pred, else_pc=end, end_pc=end, elect=False)
    else:
        c.flush()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _mods(c: PtxCtx) -> dict:
    policy = c.opt_src("cache_policy") if "cache_policy" in c.ops else None
    return pb.mem_mods(cache=CACHE_OPS.get(c.mod("cop"), "Default"), evict=EVICT.get(c.mod("l1ev"), "Normal"),
                       l2_prefetch=L2_PREFETCH.get(c.mod("prefetch"), 0), policy=policy, nc=c.flag("nc"),
                       uniform=c.name.startswith("ldu"))


def _vec_lanes(c: PtxCtx) -> int:
    token = c.mod("vec")
    return int(token[1:]) if token else 1


def lower_ld(c: PtxCtx) -> None:
    lw = c.lw
    lanes = _vec_lanes(c)
    ty = c.ptx_ty("type", lanes)
    addr_node = c.nodes("addr")[0]
    dsts = c.dsts("d")
    mods = _mods(c)
    if c.name == "ld_proxy_readonly":
        mods["nc"] = True
    sem, scope = c.sem(), c.scope()
    value = lw.builder.reg(ty)
    target = lw.buffer_access(addr_node, c.mod("space"))
    if target is not None:
        buf, offset = target
        fields = dict(ty=ty, dst=value, buf=buf, offset=offset, sem=sem, scope=scope, mods=mods)
        variant = "Load"
    else:
        addr, space = c.addr("addr")
        fields = dict(ty=ty, dst=value, addr=addr, space=space, sem=sem, scope=scope, mods=mods)
        variant = "LoadAddr"
    b = lw.builder
    if_pc = None
    if c.pred is not None:
        if_pc = b.emit("If", site=c.site(), cond=c.pred, else_pc=0, end_pc=0, elect=False)
    b.emit(variant, site=c.site(), **fields)
    lanes_out = [value] if lanes == 1 else lw.unpack(value, lanes)
    for reg, lane in zip(dsts, lanes_out):
        if reg is not None:
            lw.assign(reg, lane)
    c.flush()
    if if_pc is not None:
        end = b.emit("EndIf")
        b.patch(if_pc, "If", cond=c.pred, else_pc=end, end_pc=end, elect=False)


def lower_st(c: PtxCtx) -> None:
    lw = c.lw
    lanes = _vec_lanes(c)
    ty = c.ptx_ty("type", lanes)
    elem = ty.with_lanes(1)
    values = [lw.convert(v, elem) for v in c.srcs("value")]
    value = values[0] if lanes == 1 else lw.pack(values, ty)
    addr_node = c.nodes("addr")[0]
    mods = _mods(c)
    sem, scope = c.sem(), c.scope()
    target = lw.buffer_access(addr_node, c.mod("space"))
    if target is not None:
        buf, offset = target
        c.emit("Store", ty=ty, buf=buf, offset=offset, value=value, sem=sem, scope=scope, mods=mods)
    else:
        addr, space = c.addr("addr")
        c.emit("StoreAddr", ty=ty, addr=addr, space=space, value=value, sem=sem, scope=scope, mods=mods)


def lower_discard(c: PtxCtx) -> None:
    addr, space = c.addr("addr", "Global")
    c.emit("Discard", addr=addr, space=space, size=128)


def lower_atom(c: PtxCtx) -> None:
    lw = c.lw
    lanes = _vec_lanes(c)
    ty = c.ptx_ty("type", lanes)
    sources = c.srcs("value")
    # Pieces tile the access type exactly: `.v2.bf16x2` packs two bf16x2 pieces
    # into bf16x4 (W4's pack tiling rule), `.v4.f32` four f32 pieces.
    piece = ty.with_lanes(ty.lanes // max(len(sources), 1))
    values = [lw.convert(v, piece) for v in sources]
    value = values[0] if len(values) == 1 else lw.pack(values, ty)
    cmp = lw.convert(c.src("compare"), ty) if c.has("compare") else None
    addr, space = c.addr("addr")
    dsts = c.dsts("d") if c.has("d") else []
    result = lw.builder.reg(ty) if dsts else None
    op = ATOM_OPS[c.mod("op")]
    sem = c.sem("Relaxed")
    b = lw.builder
    # A guarded atom/red (`@p`, e.g. `pred=lane % 2 == 0` on a vector form) runs,
    # and writes its destinations, only where the guard holds (W4-12).
    if_pc = None
    if c.pred is not None:
        if_pc = b.emit("If", site=c.site(), cond=c.pred, else_pc=0, end_pc=0, elect=False)
    b.emit("Atom", site=c.site(), op=op, ty=ty, dst=result, addr=addr, space=space, value=value,
           cmp=cmp, sem=sem, scope=c.scope(), ftz=not c.flag("noftz") and ty.elem == "F32")
    if result is not None:
        outs = [result] if len(dsts) == 1 else lw.unpack(result, len(dsts))
        for reg, lane in zip(dsts, outs):
            if reg is not None:
                lw.assign(reg, lane)
    c.flush()
    if if_pc is not None:
        end = b.emit("EndIf")
        b.patch(if_pc, "If", cond=c.pred, else_pc=end, end_pc=end, elect=False)
        c.pred = None


def lower_cvta(c: PtxCtx) -> None:
    space = SPACES.get(c.mod("space"), "Generic")
    src = c.src("ptr") if "ptr" in c.ops else c.src("a")
    dst = c.dst("d")
    to_generic = c.name == "cvta_generic" or c.mod("dir") != "to"
    c.emit("Cvta", dst=dst, src=src, space=space, to_generic=to_generic if c.name != "cvta" else False)


def lower_mapa(c: PtxCtx) -> None:
    space = "SharedCluster" if c.name in ("mapa_u32", "mapa_u64_shared") or c.mod("space") else "Generic"
    c.emit("Mapa", dst=c.dst("d"), src=c.src("a"), rank=c.src("b"), space=space)


def lower_getctarank(c: PtxCtx) -> None:
    space = "SharedCluster" if c.name == "getctarank" else "Generic"
    c.emit("GetCtaRank", dst=c.dst("d"), src=c.src("a"), space=space)


def lower_isspacep(c: PtxCtx) -> None:
    c.emit("Isspacep", dst=c.dst("p"), src=c.src("a"), space=SPACES.get(c.mod("space"), "Generic"))


def _matrix_shape(token: str) -> str:
    return {"m8n8": "M8N8", "m8n16": "M8N16", "m16n8": "M16N8", "m16n16": "M16N16"}[token]


def lower_ldmatrix(c: PtxCtx) -> None:
    addr, space = c.addr("p", "Shared")
    fmt = "B16"
    if c.name == "ldmatrix_m16n16_b8":
        fmt = "B8"
    elif c.name == "ldmatrix_s8_s4":
        fmt = "S8S4"
    elif c.name == "ldmatrix_b8fmt":
        fmt = {"b6x16_p32": "B6x16P32", "b4x16_p64": "B4x16P64"}[c.mod("src_fmt")]
    dsts = [d if d is not None else c.scratch(pb.Ty("U32")) for d in c.dsts("r")]
    c.emit("LdMatrix", dsts=dsts, addr=addr, space=space, shape=_matrix_shape(c.mod("shape")),
           num=c.int_mod("num", "x"), trans=c.flag("trans"), fmt=fmt)


def lower_stmatrix(c: PtxCtx) -> None:
    addr, space = c.addr("p", "Shared")
    c.emit("StMatrix", srcs=c.srcs("r"), addr=addr, space=space, shape=_matrix_shape(c.mod("shape")),
           num=c.int_mod("num", "x"), trans=c.flag("trans"))


# ---------------------------------------------------------------------------
# Async copies / TMA
# ---------------------------------------------------------------------------


def lower_cp_async(c: PtxCtx) -> None:
    dst, _ = c.addr("dst_mem", "Shared")
    src, _ = c.addr("src_mem", "Global")
    mods = pb.mem_mods(cache=CACHE_OPS[c.mod("cop")], l2_prefetch=L2_PREFETCH.get(c.mod("prefetch"), 0),
                       policy=c.opt_src("cache_policy"))
    c.emit("CpAsync", dst=dst, src=src, cp_size=c.uimm("cp_size", 8), src_size=c.opt_src("src_size"),
           ignore_src=c.opt_src("ignore_src"), mods=mods)


def lower_async_commit(c: PtxCtx) -> None:
    c.emit("AsyncCommit", domain="Bulk" if "bulk" in c.name else "CpAsync")


def lower_async_wait(c: PtxCtx) -> None:
    domain = "Bulk" if "bulk" in c.name else "CpAsync"
    if c.name == "cp_async_wait_all":
        c.emit("AsyncCommit", domain=domain)
        c.emit("AsyncWait", domain=domain, n=0, read=False)
        return
    group = c.imm("group")
    if not 0 <= group < 1 << 32:
        raise _Unsupported(c.node, f"{c.d.op_name}: wait_group count {group} out of range")
    c.emit("AsyncWait", domain=domain, n=group, read=c.flag("read"))


def lower_cp_async_mbar_arrive(c: PtxCtx) -> None:
    mbar, space = c.mbar_addr("addr")
    c.emit("CpAsyncMbarArrive", mbar=mbar, space=space, noinc=c.flag("noinc"))


def _completion(c: PtxCtx, cluster: bool = False) -> Any:
    if "mbar" in c.ops and c.has("mbar"):
        if cluster:
            # PTX: with a .shared::cluster destination the mbarrier is the
            # destination CTA's, a .shared::cluster address (the TVM table tags
            # the operand `.shared`); a shared::cta address is a valid
            # shared::cluster address of the executing CTA.
            mbar = c.lw.address_in(c.src("mbar"), "SharedCluster")
            return pb.bulk_completion(mbar, "SharedCluster")
        mbar, space = c.addr("mbar", "Shared")
        return pb.bulk_completion(mbar, space)
    return pb.bulk_completion(None)


def _multicast(c: PtxCtx) -> pb.Operand | None:
    for name in ("cta_mask",):
        if c.has(name):
            return c.src(name)
    return None


def lower_bulk_copy(c: PtxCtx) -> None:
    dst, dst_space = c.addr("dst_mem")
    src, src_space = c.addr("src_mem")
    reduce = None
    if c.name.startswith("cp_reduce"):
        reduce = [ATOM_OPS[c.mod("redop")], pb.Ty.from_ptx(c.mod("type")).elem]
    mods = pb.mem_mods(policy=c.opt_src("cache_policy"))
    c.emit("BulkCopy", dst=dst, dst_space=dst_space, src=src, src_space=src_space, size=c.src("size"),
           # The completion barrier lives with a shared::cluster destination
           # (possibly in the peer CTA), W2-20 flash_attention_backward:1277.
           completion=_completion(c, cluster=dst_space == "SharedCluster"),
           multicast=_multicast(c), reduce=reduce,
           byte_mask=c.opt_src("byte_mask") if "byte_mask" in c.ops else None,
           ignore_oob=_ignore_oob(c), report=_report(c), mods=mods)


def _report(c: PtxCtx) -> Any:
    """Contract item 13: ``mbarrier::report::*`` copy forms -> ``ReportMode`` (None = disabled)."""
    token = c.mod("report")
    if not token or token.endswith("::disabled"):
        return None
    if "per_element" in token:
        return "PerElementFf"
    if "per_16bytes" in token:
        # `.per_16bytes::<hex>` (W2-8): the pattern and its element width
        # (hex digits x 4). The bare legacy form keeps failing closed.
        digits = token.rsplit("per_16bytes", 1)[1].lstrip(":")
        if not digits:
            return "Per16Bytes"
        if len(digits) not in (1, 2, 4, 8) or any(ch not in "0123456789abcdefABCDEF" for ch in digits):
            raise _Unsupported(c.node, f"{c.d.op_name}: report pattern {digits!r}")
        return {"Per16BytesPattern": {"pattern": int(digits, 16), "bits": 4 * len(digits)}}
    raise _Unsupported(c.node, f"{c.d.op_name}: report mode {token!r}")


def _overrides(c: PtxCtx) -> list[dict]:
    """Per-instruction tensor-map overrides (contract item 13)."""
    out: list[dict] = []
    elem_bits = 16 if c.name.endswith("_b16") else 8 if c.name.endswith("_b8") else 0
    if "global_address" in c.ops and c.has("global_address"):
        out.append({"field": "GlobalAddress", "ord": None, "value": pb.opnd(c.src("global_address")),
                    "elem_bits": 0})
    if "tensor_size" in c.ops:
        for ordinal, value in enumerate(c.srcs("tensor_size")):
            out.append({"field": "GlobalDim", "ord": ordinal, "value": pb.opnd(value), "elem_bits": elem_bits})
    # Contract item 21: per-dimension lower strides + one shared upper operand.
    if "lower_stride" in c.ops:
        for ordinal, value in enumerate(c.srcs("lower_stride")):
            out.append({"field": "GlobalStride", "ord": ordinal, "value": pb.opnd(value), "elem_bits": elem_bits})
    if "upper_stride" in c.ops and c.has("upper_stride"):
        out.append({"field": "GlobalStrideUpper", "ord": None, "value": pb.opnd(c.src("upper_stride")),
                    "elem_bits": elem_bits})
    return out


def _ignore_oob(c: PtxCtx) -> Any:
    """Contract item 24: ``.ignore_oob`` with optional left/right byte counts (null = 0)."""
    if not c.flag("ignore_oob"):
        return None
    left = c.opt_src("ignore_bytes_left") if "ignore_bytes_left" in c.ops else None
    right = c.opt_src("ignore_bytes_right") if "ignore_bytes_right" in c.ops else None
    return {"ignore_bytes_left": pb.opt_opnd(left), "ignore_bytes_right": pb.opt_opnd(right)}


def lower_bulk_prefetch(c: PtxCtx) -> None:
    lower_generic_ordering(c)


# `cp.reduce.async.bulk.tensor` .redOp x TensorMap element type (PTX ISA;
# legacy tensor_map.rs `RawTmaReductionOp::resolve`); and/or/xor take any
# 32- or 64-bit element type.
_TMA_REDUCE_DTYPES = {
    "add": {"U32", "S32", "U64", "F32", "TF32", "F16", "BF16"},
    "min": {"U32", "S32", "U64", "S64", "F16", "BF16"},
    "max": {"U32", "S32", "U64", "S64", "F16", "BF16"},
    "inc": {"U32"},
    "dec": {"U32"},
}


def _static_tmap_spec(c: PtxCtx) -> Any:
    """The host-encoded TensorMapSpec the ``tmap`` operand names, if static."""
    from .memory import handle

    stack = list(c.nodes("tmap"))
    while stack:
        node = stack.pop()
        if type_key(node) == "ir.Var":
            ref = c.lw.refs.get(handle(node))
            if ref is None:
                continue
            program = c.lw.builder.program
            slot = program.buffers[ref.buf].param_slot
            if slot is None:
                return None
            decl = program.host_abi[slot]
            return getattr(decl, "tensor_map", None)
        stack.extend(getattr(node, "args", ()) or ())
        if hasattr(node, "buffer"):
            stack.append(node.buffer.data)
    return None


def _check_tma_reduce_dtype(c: PtxCtx, redop: str) -> None:
    spec = _static_tmap_spec(c)
    if spec is None or spec.force_cu_dtype is not None:
        return  # runtime check (numsim-core oplib `tma_reduce_valid`)
    dtype = spec.dtype
    allowed = _TMA_REDUCE_DTYPES.get(redop)
    ok = pb._DTYPE_BITS.get(dtype) in (32, 64) if allowed is None else dtype in allowed
    if not ok:
        raise _Unsupported(c.node, f"cp.reduce.async.bulk.tensor operation .{redop} is invalid for TensorMap dtype {dtype}")


def lower_tma(c: PtxCtx) -> None:
    name = c.name
    if name.startswith("cp_reduce_async_bulk_tensor"):
        direction = {"Reduce": ATOM_OPS[c.mod("redop")]}
        _check_tma_reduce_dtype(c, c.mod("redop"))
    elif "prefetch" in name:
        direction = "Prefetch"
    elif "s2g" in name:
        direction = "Store"
    else:
        direction = "Load"
    mode_token = c.mod("load_mode")
    mode = {"": "Tile", "tile": "Tile", "tile::gather4": "TileGather4", "tile::scatter4": "TileScatter4",
            "im2col": "Im2col", "im2col::w": "Im2colW", "im2col::w::128": "Im2colW128",
            "im2col_no_offs": "Im2colNoOffs",
            # W4-17: the wide no-offsets store/reduce plans as Im2colW (the
            # wide layout; stores take no offsets either way).
            "im2col_no_offs::w": "Im2colW"}.get(mode_token)
    if mode is None and "no_offs_w" in name:
        mode = "Im2colNoOffs"
    if mode is None:
        raise _Unsupported(c.node, f"{c.d.op_name}: load mode {mode_token!r}")
    # PTX: `tensorMap` is the *generic* address of a map in .param, .const or
    # .global space (V2C-9). The TVM operand table tags it `.global`, but a
    # `__grid_constant__` map's address (AddrOf of a Param buffer) lies in the
    # param aperture, so the access space is always Generic.
    tmap, tmap_space = c.src("tmap"), "Generic"
    if c.lw.operand_ty(tmap).bits == 32:
        tmap, tmap_space = c.addr("tmap", "Shared")  # u32 window offset of a map in smem
    coords = c.srcs("coords")
    offsets = c.srcs("im2col_info") if c.has("im2col_info") else []
    if direction in ("Load",):
        smem, smem_space = c.addr("dst_mem")
    elif direction == "Prefetch":
        smem, smem_space = c.lw.const("uint32", 0), "Shared"
    else:
        smem, smem_space = c.addr("src_mem")
    cta_group = c.int_mod("cta_group", "cta_group::")
    c.emit("Tma", dir=direction, mode=mode, tmap=tmap, tmap_space=tmap_space, coords=coords,
           im2col_offsets=offsets, smem=smem, smem_space=smem_space, completion=_completion(c),
           multicast=_multicast(c), cta_group=cta_group, overrides=_overrides(c), report=_report(c),
           mods=pb.mem_mods(policy=c.opt_src("cache_policy")))


def lower_st_async(c: PtxCtx) -> None:
    lanes = _vec_lanes(c)
    ty = c.ptx_ty("type", lanes)
    values = [c.lw.convert(v, ty.with_lanes(1)) for v in c.srcs("b" if "b" in c.ops else "value")]
    value = values[0] if lanes == 1 else c.lw.pack(values, ty)
    release = c.name.endswith("_release")
    addr, _ = c.addr("addr", "Global" if release else "SharedCluster")
    # Contract item 12: the .release forms have no completion mbarrier.
    mbar = None if release else c.addr("mbar", "SharedCluster")[0]
    red = ATOM_OPS[c.mod("op")] if c.name.startswith("red_async") else None
    c.emit("StAsync", ty=ty, value=value, addr=addr, mbar=mbar, red=red, sem=c.sem("Weak"),
           scope=c.scope("Cluster"))


def lower_st_bulk(c: PtxCtx) -> None:
    addr, space = c.addr("addr", "Shared")
    c.emit("StBulk", addr=addr, space=space, size=c.src("size"))


def lower_tensormap_replace(c: PtxCtx) -> None:
    field = {"global_address": "GlobalAddress", "rank": "Rank", "box_dim": "BoxDim", "global_dim": "GlobalDim",
             "global_stride": "GlobalStride", "element_stride": "ElementStride", "elemtype": "ElemType",
             "interleave_layout": "InterleaveLayout", "swizzle_mode": "SwizzleMode",
             "fill_mode": "FillMode"}[c.mod("field")]
    # No state-space qualifier = generic addressing (PTX); a descriptor image
    # in shared memory then resolves through the generic shared window.
    addr, space = c.addr("addr")
    ordinal = c.uimm("ord", 8) if "ord" in c.ops else None
    value = c.src("new_val")
    c.emit("TensorMapReplace", tmap=addr, space=space, field=field, ord=ordinal, value=value)


def lower_tensormap_cp_fence(c: PtxCtx) -> None:
    dst, _ = c.addr("dst_mem", "Global")
    src, _ = c.addr("src_mem", "Shared")
    c.emit("TensorMapCopyFence", dst=dst, src=src, size=128, scope=c.scope())


def lower_generic_ordering(c: PtxCtx) -> None:
    """Ordering-only hints (prefetch/applypriority): a ``Ptx`` op with no effects."""
    lower_generic_any(c)


def lower_generic_any(c: PtxCtx) -> None:
    lw = c.lw
    srcs: list[pb.Operand] = []
    for info, values in zip(c.d.operands, c.d.values):
        if info.literal is not None:
            continue
        if info.rw != "r":
            raise _Unsupported(c.node, f"{c.d.op_name}: unexpected destination")
        srcs.extend(lw.expr(v) for v in values)
    op = lw.builder.op(pb.OpKey(c.d.op_name, c.d.mod_tokens))
    lw.builder.emit("Ptx", site=c.site(), op=op, dsts=[], srcs=srcs, pred=c.pred, keep_dst=False)


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------


def lower_bar(c: PtxCtx) -> None:
    lw = c.lw
    aligned = c.flag("aligned") or c.name.startswith("bar_")
    ident = c.src("a")
    count = c.opt_src("b")
    action = c.mod("action")
    if action == "arrive":
        kind: Any = "Arrive"
    elif action == "sync":
        kind = "Sync"
    else:
        op = {"popc": "Popc", "and": "And", "or": "Or"}[c.mod("op")]
        dst = c.dst("d")
        kind = {"Red": {"op": op, "pred": pb.opnd(lw.cast_to(c.src("c"), pb.Ty("Pred"))), "dst": dst}}
    c.emit("Barrier", kind=kind, id=ident, count=count, aligned=aligned)


def lower_bar_warp_sync(c: PtxCtx) -> None:
    c.emit("WarpSync", membermask=c.src("membermask"))


def lower_cluster_barrier(c: PtxCtx) -> None:
    if c.mod("action") == "arrive":
        c.emit("ClusterArrive", sem=c.sem("Release"), aligned=c.flag("aligned"))
    else:
        c.emit("ClusterWait", acquire=True, aligned=c.flag("aligned"))


def lower_fence(c: PtxCtx) -> None:
    name = c.name
    if name == "fence":
        # PTX ISA (fence): "If the optional .sem qualifier is absent, .acq_rel is
        # assumed by default." (W5-16; racecheck-isa-answers.md R10)
        c.emit("Fence", kind="Thread", sem=c.sem("AcqRel"), scope=c.scope())
    elif name == "fence_mbarrier_init":
        c.emit("Fence", kind="MbarrierInit", sem="Release", scope="Cluster")
    elif name == "fence_proxy":
        if c.mod("proxykind") == "alias":
            c.emit("Fence", kind="ProxyAlias", sem="Weak", scope="Cta")
        else:
            token = c.mod("space")
            c.emit("Fence", kind={"ProxyAsync": SPACES[token] if token else None}, sem="Weak", scope="Cta")
    elif name == "fence_proxy_tensormap_release":
        c.emit("Fence", kind="TensormapRelease", sem="Release", scope=c.scope())
    elif name == "fence_proxy_tensormap_acquire":
        addr, space = c.addr("addr", "Generic")
        c.emit("Fence", kind={"TensormapAcquire": {"addr": pb.opnd(addr), "space": space}}, sem="Acquire",
               scope=c.scope())
    else:
        raise _Unsupported(c.node, f"{c.d.op_name}: fence kind")


def lower_mbarrier(c: PtxCtx) -> None:
    name = c.name
    action = c.mod("action")
    if name == "mbarrier_init":
        mbar, space = c.mbar_addr("addr")
        c.emit("MbarInit", mbar=mbar, space=space, count=c.src("count"),
               layout_v1=c.mod("layout") == "layout::v1")
    elif name == "mbarrier_inval":
        mbar, space = c.mbar_addr("addr")
        c.emit("MbarInval", mbar=mbar, space=space)
    elif action in ("arrive", "arrive_drop"):
        mbar, space = c.mbar_addr("addr")
        count = c.opt_src("count") if "count" in c.ops else None
        expect = None
        if c.flag("expect_tx"):
            expect = c.opt_src("tx_count") if "tx_count" in c.ops else count
            if "tx_count" not in c.ops:
                count = None
        state = c.dst("state") if c.has("state") and c.info("state").rw != "r" else None
        c.emit("MbarArrive", mbar=mbar, space=space, count=count, expect_tx=expect, drop=action == "arrive_drop",
               no_complete=c.flag("nocomplete"), sem=c.sem("Release"), scope=c.scope("Cta"),
               multicast=_multicast(c), state=state)
    elif action in ("expect_tx", "complete_tx"):
        mbar, space = c.mbar_addr("addr")
        c.emit("MbarTx", op="Expect" if action == "expect_tx" else "Complete", mbar=mbar, space=space,
               bytes=c.src("tx_count"), multicast=_multicast(c), scope=c.scope("Cta"))
    elif action in ("test_wait", "try_wait"):
        mbar, space = c.mbar_addr("addr")
        if c.flag("parity"):
            phase = pb.phase_parity(c.src("phase"))
        else:
            phase = pb.phase_state(c.src("state" if "state" in c.ops else "phase"))
        dst = c.dst("wait_complete")
        # Contract item 11: report forms read the report predicate/value from the same snapshot.
        report = c.dst("report_predicate") if "report_predicate" in c.ops else None
        report_value = c.dst("report_value") if "report_value" in c.ops else None
        c.emit("MbarTestWait", kind="Test" if action == "test_wait" else "Try", mbar=mbar, space=space,
               phase=phase, sem=c.sem("Acquire"), scope=c.scope("Cta"), dst=dst, report=report,
               report_value=report_value)
    elif name == "mbarrier_pending_count":
        state = c.src("state")
        c.emit("MbarQuery", dst=c.dst("count"), op={"PendingCount": {"state": pb.opnd(state)}})
    elif name == "mbarrier_check_layout":
        mbar, space = c.mbar_addr("addr")
        c.emit("MbarQuery", dst=c.dst("matches"), op={"CheckLayout": {"mbar": pb.opnd(mbar), "space": space,
                                                                         "layout_v1": c.mod("layout") == "layout::v1"}})
    else:
        raise _Unsupported(c.node, f"{c.d.op_name}: mbarrier action {action!r}")


def lower_setmaxnreg(c: PtxCtx) -> None:
    c.emit("SetMaxNReg", inc=c.mod("action") == "inc", count=c.uimm("nreg"))


def lower_griddepcontrol(c: PtxCtx) -> None:
    c.lw.builder.program.requirements.grid_dependency = True
    c.emit("GridDepControl", launch_dependents=c.mod("action") == "launch_dependents")


def lower_clc_try_cancel(c: PtxCtx) -> None:
    resp, _ = c.addr("addr", "Shared")
    mbar, _ = c.addr("mbar", "Shared")
    c.emit("ClcTryCancel", resp=resp, mbar=mbar, multicast=c.flag("multicast"))


# ---------------------------------------------------------------------------
# Warp collectives
# ---------------------------------------------------------------------------


def lower_shfl(c: PtxCtx) -> None:
    mode = {"up": "Up", "down": "Down", "bfly": "Bfly", "idx": "Idx"}[c.mod("mode")]
    dst = c.dst("d")
    dst_pred = c.dst("p") if "p" in c.ops else None
    c.emit("Shfl", mode=mode, ty=c.lw.builder.reg_ty(dst), dst=dst, dst_pred=dst_pred, src=c.src("a"),
           lane=c.src("b"), clamp=c.src("c"), membermask=c.src("membermask"))


def lower_vote(c: PtxCtx) -> None:
    mode = {"all": "All", "any": "Any", "uni": "Uni", "ballot": "Ballot"}[c.mod("mode")]
    pred = c.lw.cast_to(c.src("a"), pb.Ty("Pred"))
    c.emit("Vote", mode=mode, dst=c.dst("d"), pred=pred, membermask=c.src("membermask"))


def lower_redux(c: PtxCtx) -> None:
    ty = c.ptx_ty()
    src = c.lw.convert(c.src("a"), ty)
    if c.flag("abs"):
        # redux.abs reduces |a|: apply Abs before the collective.
        absolute = c.lw.builder.reg(ty)
        c.lw.builder.emit("Unary", op="Abs", ty=ty, dst=absolute, a=src)
        src = absolute
    dst = c.dst("d")
    mask = c.src("membermask")
    if not c.flag("nan"):
        c.emit("Redux", op=REDUX_OPS[c.mod("op")], ty=ty, dst=dst, src=src, membermask=mask)
        return
    # redux.NaN: canonical NaN if any member's input is NaN, else the NaN-ignoring
    # reduction. Instr::Redux has no NaN flag, so compose it from Redux + Vote.
    lw, b = c.lw, c.lw.builder
    reduced = b.reg(ty)
    is_nan = b.reg(pb.Ty("Pred"))
    any_nan = b.reg(pb.Ty("Pred"))
    b.emit("Redux", site=c.site(), op=REDUX_OPS[c.mod("op")], ty=ty, dst=reduced, src=src, membermask=mask)
    b.emit("Unary", op="IsNan", ty=ty, dst=is_nan, a=src)
    b.emit("Vote", site=c.site(), mode="Any", dst=any_nan, pred=is_nan, membermask=mask)
    result = b.reg(ty)
    b.emit("Select", ty=ty, dst=result, cond=any_nan, a=b.const(ty, 0x7FFF_FFFF), b=reduced)
    lw.assign(dst, result)
    c.flush()


def lower_elect(c: PtxCtx) -> None:
    lane = c.dsts("d")[0]
    pred = c.dst("p")
    if pred is None:
        pred = c.scratch(pb.Ty("Pred"))
    c.emit("Elect", dst_pred=pred, dst_lane=lane, membermask=c.src("membermask"))


def lower_activemask(c: PtxCtx) -> None:
    c.emit("ReadSpecial", dst=c.dst("d"), sreg="ActiveMask")


# ---------------------------------------------------------------------------
# tcgen05
# ---------------------------------------------------------------------------


def _cta_group(c: PtxCtx) -> int:
    return c.int_mod("cta_group", "cta_group::") or 1


def lower_tcgen05(c: PtxCtx) -> None:
    name = c.name
    lw = c.lw
    # Contract item 27: the PTX table forms carry the full taddr (row/column folded in
    # by the kernel, e.g. via cuda.get_tmem_addr), so the separate offsets are 0.
    zero = lw.const("int32", 0)
    if name.startswith("tcgen05_alloc"):
        dst, _ = c.addr("dst", "Shared")
        c.emit("TcgenAlloc", dst=dst, ncols=c.src("ncols"), cta_group=_cta_group(c), exclusive="exclusive" in name)
    elif name.startswith("tcgen05_dealloc"):
        c.emit("TcgenDealloc", taddr=c.src("taddr"), ncols=c.src("ncols"), cta_group=_cta_group(c),
               exclusive="exclusive" in name)
    elif name == "tcgen05_relinquish_alloc_permit":
        c.emit("TcgenRelinquish", cta_group=_cta_group(c))
    elif name.startswith("tcgen05_commit"):
        mbar, space = c.mbar_addr("mbar")
        multicast = c.src("mask") if c.has("mask") else (c.src("cta_mask") if c.has("cta_mask") else None)
        token = c.mod("multicast")
        width = int(token.rsplit("::", 1)[1].rstrip("b")) if token.endswith(("::16b", "::32b")) else None
        c.emit("TcgenCommit", mbar=mbar, space=space, cta_group=_cta_group(c), multicast=multicast,
               sync_restrict=c.flag("sync_restrict"), multicast_width=width)
    elif name == "tcgen05_wait":
        c.emit("TcgenWait", st=c.mod("action") == "wait::st")
    elif name == "tcgen05_fence":
        kind = "Tcgen05Before" if c.mod("action") == "fence::before_thread_sync" else "Tcgen05After"
        c.emit("Fence", kind=kind, sem="Weak", scope="Cta")
    elif name.startswith("tcgen05_ld"):
        shape = _tc_shape(c)
        spcompress = "spcompress" in name
        # .spcompress forms: dsts are the metadata lanes then the compressed-data lanes
        # (TVM table operand order ``mdata``, ``cdata``).
        groups = ("mdata", "cdata") if spcompress else ("r",)
        dsts = [d if d is not None else c.scratch(pb.Ty("U32")) for g in groups for d in c.dsts(g)]
        red = None
        if "redval" in c.ops:
            red_regs = [d if d is not None else c.scratch(pb.Ty.from_ptx(c.mod("type"))) for d in c.dsts("redval")]
            red = [REDUX_OPS[c.mod("redop") or c.mod("rowop")], red_regs]
        elif spcompress:
            # W4-17: the compression's max/min selection rides in `red` with
            # no reduction destinations (`.abs` in `red_abs`).
            red = [REDUX_OPS[c.mod("rowop")], []]
        taddr = c.src("taddr")
        c.emit("TcgenLd", dsts=dsts, taddr=taddr, row=zero, col=zero, shape=shape, num=c.int_mod("num", "x"),
               pack=c.flag("pack"), red=red, red_abs=c.flag("abs"), red_nan=c.flag("nan"), spcompress=spcompress)
    elif name.startswith("tcgen05_st"):
        shape = _tc_shape(c)
        c.emit("TcgenSt", srcs=c.srcs("r"), taddr=c.src("taddr"), row=zero, col=zero, shape=shape, num=c.int_mod("num", "x"),
               unpack=c.flag("unpack"))
    elif name == "tcgen05_cp":
        rows, bits = (int(x) for x in c.mod("shape").rstrip("b").split("x"))
        multicast = {"": 0, "warpx2::02_13": 1, "warpx2::01_23": 2, "warpx4": 3}[c.mod("multicast")]
        decompress = {"": 0, "b6x16_p32": 6, "b4x16_p64": 4}[c.mod("src_fmt")]
        c.emit("TcgenCp", taddr=c.src("taddr"), row=zero, col=zero, sdesc=c.src("s_desc"), rows=rows, bits=bits, multicast=multicast,
               decompress_bits=decompress, cta_group=_cta_group(c))
    elif name.startswith("tcgen05_mma"):
        lower_tcgen05_mma(c)
    else:
        raise _Unsupported(c.node, f"{c.d.op_name}: tcgen05 form")
    del lw


def _tc_shape(c: PtxCtx) -> Any:
    token = c.mod("shape")
    if token == "16x32bx2":
        return {"S16x32bx2": {"split_off": c.uimm("imm_half_splitoff")}}
    return TC_SHAPES[token]


def _collector(token: str) -> tuple[str, int]:
    if not token:
        return "None", 0
    parts = token.split("::")          # collector::a::fill / collector::b0::use
    which, op = parts[1], parts[2]
    buffer = int(which[1:]) if len(which) > 1 else 0
    return COLLECTOR[op], buffer


def lower_tcgen05_mma(c: PtxCtx) -> None:
    name = c.name
    lut_b = "lut_b" in name
    kind = MMA_KINDS.get(c.mod("kind"))
    if kind is None:
        raise _Unsupported(c.node, f"{c.d.op_name}: MMA {c.mod('kind')} is not in TcMmaKind")
    if c.mod("block_size") == "block16" or c.mod("scale_vec") == "scale_vec::4X":
        block = 16
    else:
        block = 32
    a = {"Smem": pb.opnd(c.src("a_desc"))} if "a_desc" in c.ops else {"Tmem": pb.opnd(c.src("a_tmem"))}
    block_scale = None
    if "sfa_tmem" in c.ops:
        block_scale = [pb.opnd(c.src("sfa_tmem")), pb.opnd(c.src("sfb_tmem")), block]
    lanes = c.srcs("disable_output_lane") if "disable_output_lane" in c.ops else []
    if "zero_col_mask" in c.ops and c.has("zero_col_mask"):
        lanes = [c.src("zero_col_mask")]
    collector_a, _ = _collector(c.mod("collector_a"))
    collector_b, b_buffer = _collector(c.mod("collector_b"))
    b_desc = c.src("b_compressed_desc") if lut_b else c.src("b_desc")
    # Contract item 18: the lut_b table operand (``b_decompress_metadata``; a TMEM
    # address in TVM's PTX table) travels in ``lut_b_addr``.
    lut_b_addr = c.src("b_decompress_metadata") if lut_b else None
    if kind == "F8f6f4" and _cta_group(c) == 2 and c.lw.builder.program.arch not in TCGEN_DESCRIPTOR_ARCHES:
        # The CTA-pair f8f6f4 descriptor semantics differ between SM100/SM103
        # and SM107 (legacy require_tcgen_descriptor_layout): the kernel must
        # name its exact architecture.
        raise _Unsupported(c.node, f"{c.d.op_name}: kind::f8f6f4 cta_group::2 requires tirx.cuda_arch in "
                                   f"{sorted(TCGEN_DESCRIPTOR_ARCHES)}, got {c.lw.builder.program.arch!r}")
    c.emit("TcgenMma", kind=kind, cta_group=_cta_group(c), d=c.src("d_tmem"), a=a, b_desc=b_desc,
           idesc=c.src("idesc"), enable_input_d=c.lw.cast_to(c.src("enable_input_d"), pb.Ty("Pred")),
           ws=c.flag("ws"), ws_b_buffer=b_buffer, block_scale=block_scale, scale_input_d=None,
           sparse_meta=c.src("sp_meta_tmem") if "sp_meta_tmem" in c.ops else None,
           disable_output_lane=lanes, collector_a=collector_a, collector_b=collector_b,
           ashift=c.flag("ashift"), lut_b=lut_b, lut_b_addr=lut_b_addr)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_EXACT: dict[str, Callable[[PtxCtx], None]] = {
    "ld": lower_ld, "ld_vec": lower_ld, "ld_vec256": lower_ld, "ldu": lower_ld, "ldu_vec": lower_ld,
    "ld_proxy_readonly": lower_ld,
    "st": lower_st, "st_vec": lower_st, "st_vec256": lower_st,
    "discard": lower_discard,
    "cvta": lower_cvta, "cvta_generic": lower_cvta, "isspacep": lower_isspacep,
    "getctarank": lower_getctarank, "getctarank_generic": lower_getctarank,
    "cp_async_commit_group": lower_async_commit, "cp_async_bulk_commit_group": lower_async_commit,
    "cp_async_wait_group": lower_async_wait, "cp_async_wait_all": lower_async_wait,
    "cp_async_bulk_wait_group": lower_async_wait,
    "cp_async_mbarrier_arrive": lower_cp_async_mbar_arrive,
    "st_bulk": lower_st_bulk, "tensormap_cp_fenceproxy": lower_tensormap_cp_fence,
    "bar_warp_sync": lower_bar_warp_sync, "setmaxnreg": lower_setmaxnreg,
    "griddepcontrol": lower_griddepcontrol, "clusterlaunchcontrol_try_cancel": lower_clc_try_cancel,
    "elect_sync": lower_elect, "activemask": lower_activemask,
    "barrier_cluster_arrive": lower_cluster_barrier, "barrier_cluster_wait": lower_cluster_barrier,
}

_PREFIX: tuple[tuple[str, Callable[[PtxCtx], None]], ...] = (
    ("atom", lower_atom),
    ("redux", lower_redux),
    ("red_async", lower_st_async),
    ("red", lower_atom),
    ("mapa", lower_mapa),
    ("ldmatrix", lower_ldmatrix),
    ("stmatrix", lower_stmatrix),
    ("cp_async_bulk_tensor", lower_tma),
    ("cp_reduce_async_bulk_tensor", lower_tma),
    ("cp_async_bulk_prefetch", lower_bulk_prefetch),
    ("cp_async_bulk", lower_bulk_copy),
    ("cp_reduce_async_bulk", lower_bulk_copy),
    ("cp_async_c", lower_cp_async),
    ("st_async", lower_st_async),
    ("tensormap_replace", lower_tensormap_replace),
    ("bar_", lower_bar),
    ("barrier_", lower_bar),
    ("fence", lower_fence),
    ("mbarrier", lower_mbarrier),
    ("shfl", lower_shfl),
    ("vote", lower_vote),
    ("redux", lower_redux),
    ("tcgen05", lower_tcgen05),
    ("prefetch", lower_generic_ordering),
    ("applypriority", lower_generic_ordering),
)

REJECTED = ("wgmma", "multimem", "fabric", "fence_proxy_fabric")


def handler_for(table_name: str) -> Callable[[PtxCtx], None]:
    if table_name.startswith(REJECTED):
        raise KeyError(table_name)
    exact = _EXACT.get(table_name)
    if exact is not None:
        return exact
    for prefix, handler in _PREFIX:
        if table_name.startswith(prefix):
            return handler
    return lower_generic


def lower_ptx(lw: "Lowerer", node: Any) -> None:
    try:
        decoded = ptx_decode.decode(node)
    except ptx_decode.PtxDecodeError as error:
        raise _Unsupported(node, str(error)) from error
    try:
        handler = handler_for(decoded.table_name)
    except KeyError:
        raise _Unsupported(node, f"{decoded.op_name} is rejected (not modeled for the SM100 target)") from None
    register_pairs = _check_distinct_registers(lw, node, decoded) \
        if decoded.table_name.startswith("spdecompress") else []
    ctx = PtxCtx(lw, node, decoded)
    b = lw.builder
    if_pc = None
    if ctx.pred is not None:
        bridge_guarded_pred_destinations(ctx)
        # The guard covers the whole instruction, operands included: a
        # predicated-off lane evaluates no memory operand (legacy; e.g.
        # `@p ld [A + Select(p, i, size)]` must not read A[size]). Inside the
        # If the op runs unguarded, so its destinations keep their values off.
        if_pc = b.emit("If", site=ctx.site(), cond=ctx.pred, else_pc=0, end_pc=0, elect=False)
        guard, ctx.pred = ctx.pred, None
    start = len(b.code)
    try:
        if register_pairs:
            emit_register_checks(lw, node, decoded, register_pairs)
        handler(ctx)
    except (pb.UnrepresentableType, _Unsupported) as error:
        if if_pc is not None:
            del b.code[if_pc:]
            del b.code_sites[if_pc:]
        if isinstance(error, _Unsupported):
            raise
        raise _Unsupported(node, f"{decoded.op_name}: {error}") from error
    if if_pc is not None:
        end = b.emit("EndIf")
        b.patch(if_pc, "If", cond=guard, else_pc=end, end_pc=end, elect=False)
    del start


def _check_distinct_registers(lw: "Lowerer", node: Any, decoded: ptx_decode.DecodedPtx) -> list:
    """spdecompress register operands must be distinct physical registers (PTX:
    overlapping data/metadata registers are undefined; legacy rejected them).

    Only register-backed operands (promoted locals, ``local`` buffers) name
    physical registers; a global/shared element operand is copied through its
    own register. Two operands over the same register root are rejected at
    transpile only when they provably overlap; disjointness proven from the
    index bounds needs nothing, and an unproven pair is returned for a run-time
    ``Assert`` on the concrete byte ranges (emitted by ``emit_register_checks``).
    """
    from tvm.sym.analyzer import Analyzer

    from .memory import MemRef, RegArray, sum_bases

    analyzer = Analyzer()
    import tvm

    for loop in lw.loop_scopes:
        try:
            analyzer.bind(loop.loop_var, tvm.ir.Range.from_min_extent(loop.min, loop.extent))
        except Exception:  # noqa: BLE001 - an unbindable range leaves the var unbounded
            pass
    operands: list[tuple] = []
    for info, values in zip(decoded.operands, decoded.values):
        for value in values:
            if value is ptx_decode.SINK or isinstance(value, str) or type_key(value) != "ir.TensorLoad":
                continue
            ref = lw.ref_of(value.source)
            if isinstance(ref, RegArray):
                root, base, elem_bytes = ("regs", id(ref.regs[0])), 0, 1
            elif isinstance(ref, MemRef) and ref.space in ("Local", "Reg"):
                buffers = lw.builder.program.buffers
                root_buf = ref.buf
                while buffers[root_buf].view_of is not None:
                    root_buf = buffers[root_buf].view_of
                root, base = ("buf", root_buf), sum_bases(buffers, ref.buf)
                elem_bytes = max(1, buffers[ref.buf].dtype.bits // 8)
            else:
                continue
            shape = ref.info.static_shape
            flat = None
            if shape is not None and not ref.info.strides and getattr(ref.info, "layout", None) is None:
                flat = 0
                for extent, index in zip(shape, value.indices):
                    flat = flat * extent + index
                flat = analyzer.simplify(flat * elem_bytes + base) if not isinstance(flat, int) \
                    else flat * elem_bytes + base
            operands.append((root, flat, elem_bytes, ref, value.indices, info.name))

    def bounds(flat: Any) -> tuple[int, int] | None:
        if isinstance(flat, int):
            return flat, flat
        if type_key(flat) == "ir.IntImm":
            return int(flat.value), int(flat.value)
        bound = analyzer.const_int_bound(flat)
        if abs(bound.min_value) >= 1 << 62 or abs(bound.max_value) >= 1 << 62:
            return None
        return int(bound.min_value), int(bound.max_value)

    runtime: list = []
    for i, (root_a, flat_a, size_a, ref_a, idx_a, name_a) in enumerate(operands):
        for root_b, flat_b, size_b, ref_b, idx_b, name_b in operands[:i]:
            if root_a != root_b:
                continue
            if flat_a is not None and flat_b is not None:
                diff = bounds(analyzer.simplify(flat_a - flat_b)) if not (
                    isinstance(flat_a, int) and isinstance(flat_b, int)) else (flat_a - flat_b,) * 2
                if diff is not None and diff[0] == diff[1] and -size_a < diff[0] < size_b:
                    raise _Unsupported(node, f"{decoded.op_name}: undefined register overlap: {name_a} and "
                                             f"{name_b} name the same physical register (aliased)")
                range_a, range_b = bounds(flat_a), bounds(flat_b)
                if range_a is not None and range_b is not None and (
                        range_a[1] + size_a <= range_b[0] or range_b[1] + size_b <= range_a[0]):
                    continue   # disjoint for every index value
                if diff is not None and (diff[0] >= size_b or diff[1] <= -size_a):
                    continue   # constant-sign separation
            runtime.append(((ref_a, idx_a, size_a), (ref_b, idx_b, size_b), f"{name_a} and {name_b}"))
    return runtime


def emit_register_checks(lw: "Lowerer", node: Any, decoded: ptx_decode.DecodedPtx, pairs: list) -> None:
    """Run-time ``Assert`` that each unproven operand pair names disjoint registers."""
    from .memory import MemRef, sum_bases

    i32 = pb.Ty("S32")

    def byte_range(ref: Any, indices: Any, size: int) -> tuple[pb.Operand, pb.Operand]:
        offset, _ = lw.flat_offset(ref, indices)
        start = lw.binary("Mul", i32, lw.cast_to(offset, i32), lw.const("int32", size))
        if isinstance(ref, MemRef):
            base = sum_bases(lw.builder.program.buffers, ref.buf)
            if base:
                start = lw.binary("Add", i32, start, lw.const("int32", base))
        return start, lw.binary("Add", i32, start, lw.const("int32", size))

    for (ref_a, idx_a, size_a), (ref_b, idx_b, size_b), names in pairs:
        start_a, end_a = byte_range(ref_a, idx_a, size_a)
        start_b, end_b = byte_range(ref_b, idx_b, size_b)
        before, after = lw.builder.reg(pb.Ty("Pred")), lw.builder.reg(pb.Ty("Pred"))
        lw.builder.emit("Compare", op="Le", ty=i32, dst=before, a=end_a, b=start_b)
        lw.builder.emit("Compare", op="Le", ty=i32, dst=after, a=end_b, b=start_a)
        ok = lw.binary("Or", pb.Ty("Pred"), before, after)
        lw.builder.emit("Assert", site=lw.site(node), cond=ok, msg=lw.builder.string(
            f"{decoded.op_name}: undefined register overlap: {names} name the same physical register"))


__all__ = ["lower_ptx", "handler_for"]
