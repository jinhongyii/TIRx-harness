"""Definite-initialization pre-pass for register-promotable locals (V2C-19/20).

Promoted locals become plain registers, which carry no validity, so a read
before any write would go unreported. This pass finds the local buffers whose
elements *may* be read before they are written along some path; lowering keeps
those in tracked memory (``TRACKED_SPACE``) and promotes the rest.

The analysis is a forward "definitely written elements" dataflow over the TIR
statement tree:

- ``If``: the two branches are intersected (no else = the state before).
- Loops: the body is analysed once from the state before the loop, which is
  the first iteration; later iterations only see more writes. A loop with a
  small constant extent and no ``break`` is unrolled so that indices become
  constants. After a loop the state is the body's state only for such
  unrolled loops; otherwise it is the state before the loop.
- An ``AllocBuffer`` inside a loop resets its buffer each iteration.
- Writes: ``BufferStore``, PTX table destinations (``w``; ``rw`` reads first;
  a predicated op may not write), helper out-parameters (``o``/``x`` roles
  through ``address_of``) and the ``wait_until`` destination.
- A read through an index the pass cannot evaluate needs every element
  written. Anything unrecognised counts as a read and never as a write, so
  the pass errs towards tracking.
"""

from __future__ import annotations

from typing import Any

from . import builtins, ptx_decode
from .dtypes import type_key

# Engine space for tracked locals. `Space::Reg` buffers are not bound by the
# scheduler yet (W2-16), so tracked locals use per-lane `Local` memory, which
# starts `Init::Uninit`.
TRACKED_SPACE = "Local"

_UNROLL_LIMIT = 64
_ALL = None  # sentinel: every element written


def _handle(node: Any) -> int:
    return int(node.__chandle__())


class _Analysis:
    def __init__(self, candidates: dict[int, tuple[int, ...]]):
        self.shapes = candidates
        self.maybe_uninit: set[int] = set()
        self.pinned: set[int] = set()  # must stay registers (wait_until destinations)

    # -- index evaluation ---------------------------------------------------
    def value(self, node: Any, env: dict[int, int]) -> int | None:
        kind = type_key(node)
        if kind == "ir.IntImm":
            return int(node.value)
        if kind == "ir.Var":
            return env.get(_handle(node))
        if kind == "ir.Cast":
            return self.value(node.value, env)
        ops = {"prim.Add": lambda a, b: a + b, "prim.Sub": lambda a, b: a - b,
               "prim.Mul": lambda a, b: a * b,
               "prim.FloorDiv": lambda a, b: a // b if b else None,
               "prim.FloorMod": lambda a, b: a % b if b else None}
        if kind in ops:
            a, b = self.value(node.a, env), self.value(node.b, env)
            return None if a is None or b is None else ops[kind](a, b)
        return None

    def flat(self, buf: int, indices: Any, env: dict[int, int]) -> int | None:
        shape = self.shapes[buf]
        if len(indices) != len(shape):
            return None
        flat = 0
        for extent, index in zip(shape, indices):
            v = self.value(index, env)
            if v is None or not 0 <= v < extent:
                return None
            flat = flat * extent + v
        return flat

    # -- state helpers ------------------------------------------------------
    def written(self, state: dict[int, Any], buf: int, flat: int | None) -> bool:
        have = state.get(buf, set())
        if have is _ALL:
            return True
        if flat is None:
            return False
        return flat in have

    def write(self, state: dict[int, Any], buf: int, flat: int | None) -> None:
        if flat is None:
            return
        have = state.setdefault(buf, set())
        if have is _ALL:
            return
        have.add(flat)
        numel = 1
        for extent in self.shapes[buf]:
            numel *= extent
        if len(have) >= numel:
            state[buf] = _ALL

    @staticmethod
    def copy(state: dict[int, Any]) -> dict[int, Any]:
        return {k: (v if v is _ALL else set(v)) for k, v in state.items()}

    @staticmethod
    def meet(a: dict[int, Any], b: dict[int, Any]) -> dict[int, Any]:
        out: dict[int, Any] = {}
        for key in a.keys() & b.keys():
            x, y = a[key], b[key]
            out[key] = y if x is _ALL else (x if y is _ALL else x & y)
        return out

    # -- reads --------------------------------------------------------------
    def reads(self, node: Any, state: dict[int, Any], env: dict[int, int], skip: Any = None) -> None:
        """Every candidate element ``node`` reads must be written already."""
        from tvm_ffi import structural_visit
        import tvm

        def on_load(load: Any, visitor: Any) -> None:
            if load is not skip:
                buf = _handle(load.source)
                if buf in self.shapes and not self.written(state, buf, self.flat(buf, load.indices, env)):
                    self.maybe_uninit.add(buf)
            for index in load.indices:
                visitor.visit(index)

        if node is None:
            return
        structural_visit(node, [(tvm.ir.TensorLoad, on_load)])

    # -- statements ---------------------------------------------------------
    def stmt(self, node: Any, state: dict[int, Any], env: dict[int, int]) -> tuple[dict[int, Any], bool]:
        """State after ``node`` and whether it may ``break``."""
        kind = type_key(node)
        if kind == "tirx.SeqStmt":
            breaks = False
            for child in node.seq:
                state, b = self.stmt(child, state, env)
                breaks = breaks or b
            return state, breaks
        if kind == "tirx.AttrStmt":
            self.reads(node.value, state, env)
            return self.stmt(node.body, state, env)
        if kind == "tirx.AllocBuffer":
            buf = _handle(node.buffer)
            state = self.copy(state)
            state.pop(buf, None)
            return state, False
        if kind == "tirx.BufferStore":
            self.reads(node.value, state, env)
            for index in node.indices:
                self.reads(index, state, env)
            buf = _handle(node.buffer)
            if buf in self.shapes:
                state = self.copy(state)
                self.write(state, buf, self.flat(buf, node.indices, env))
            return state, False
        if kind == "tirx.IfThenElse":
            self.reads(node.condition, state, env)
            then_state, b1 = self.stmt(node.then_case, self.copy(state), env)
            if node.else_case is None:
                return self.meet(then_state, state), b1
            else_state, b2 = self.stmt(node.else_case, self.copy(state), env)
            return self.meet(then_state, else_state), b1 or b2
        if kind == "tirx.For":
            return self.loop(node, state, env), False
        if kind == "tirx.While":
            self.reads(node.condition, state, env)
            self.stmt(node.body, self.copy(state), env)
            return state, False
        if kind == "tirx.Break":
            return state, True
        if kind == "tirx.Evaluate":
            return self.evaluate(node.value, state, env), False
        if kind == "tirx.Bind":
            self.reads(node.value, state, env)
            return state, False
        # DeclBuffer, ScopeIdDefStmt, AssertStmt, Continue, Return, ...: reads only.
        self.reads(node, state, env)
        return state, False

    def loop(self, node: Any, state: dict[int, Any], env: dict[int, int]) -> dict[int, Any]:
        self.reads(node.min, state, env)
        self.reads(node.extent, state, env)
        start, extent = self.value(node.min, env), self.value(node.extent, env)
        step = 1 if node.step is None else self.value(node.step, env)
        var = _handle(node.loop_var)
        if start is not None and extent is not None and step and 0 < extent <= _UNROLL_LIMIT:
            current, breaks = state, False
            for i in range(start, start + extent, step):
                current, b = self.stmt(node.body, self.copy(current), {**env, var: i})
                breaks = breaks or b
            return state if breaks else current
        inner = {k: v for k, v in env.items() if k != var}
        self.stmt(node.body, self.copy(state), inner)
        return state

    def evaluate(self, value: Any, state: dict[int, Any], env: dict[int, int]) -> dict[int, Any]:
        if type_key(value) != "ir.Call":
            self.reads(value, state, env)
            return state
        name = str(getattr(value.op, "name", ""))
        if name.startswith("tirx.ptx.") and ptx_decode.is_table_op(name):
            try:
                decoded = ptx_decode.decode(value)
            except ptx_decode.PtxDecodeError:
                self.reads(value, state, env)
                return state
            writes: list[Any] = []
            for info, nodes in zip(decoded.operands, decoded.values):
                for item in nodes:
                    if isinstance(item, str) or item is ptx_decode.SINK:
                        continue
                    if info.rw == "w" and type_key(item) == "ir.TensorLoad":
                        for index in item.indices:
                            self.reads(index, state, env)
                        writes.append(item)
                    else:
                        self.reads(item, state, env)
            self.reads(decoded.predicate, state, env)
            if decoded.predicate is not None or decoded.preserve_dst:
                return state
            return self.store_all(writes, state, env)
        helper = builtins.HELPERS.get(name)
        if name == "tirx.cuda.wait_until" and value.args:
            # The destination is loaded before the predicate (which reads it) runs.
            if type_key(value.args[0]) == "ir.TensorLoad":
                self.pinned.add(_handle(value.args[0].source))
            self.reads(value.args[1], state, env)
            state = self.store_all([value.args[0]], state, env)
            for arg in value.args[2:]:
                self.reads(arg, state, env)
            return state
        if helper is not None and any(r in "ox" for r in helper.roles.rstrip("*")):
            writes = []
            for position, arg in enumerate(value.args):
                role = builtins.role(helper.roles, position)
                target = arg.args[0] if (type_key(arg) == "ir.Call"
                                         and str(getattr(arg.op, "name", "")) == "tirx.address_of") else None
                if role in "ox" and target is not None and type_key(target) == "ir.TensorLoad":
                    if role == "x":
                        self.reads(target, state, env)
                    else:
                        for index in target.indices:
                            self.reads(index, state, env)
                    writes.append(target)
                else:
                    self.reads(arg, state, env)
            return self.store_all(writes, state, env)
        self.reads(value, state, env)
        return state

    def store_all(self, loads: list[Any], state: dict[int, Any], env: dict[int, int]) -> dict[int, Any]:
        state = self.copy(state)
        for load in loads:
            if type_key(load) != "ir.TensorLoad":
                continue
            buf = _handle(load.source)
            if buf in self.shapes:
                self.write(state, buf, self.flat(buf, load.indices, env))
        return state


def maybe_uninit_locals(statements: list[Any], candidates: dict[int, tuple[int, ...]]) -> set[int]:
    """Buffer handles (of ``candidates``: handle -> static shape) possibly read before written."""
    if not candidates:
        return set()
    analysis = _Analysis(candidates)
    state: dict[int, Any] = {}
    for statement in statements:
        state, _ = analysis.stmt(statement, state, {})
    return analysis.maybe_uninit - analysis.pinned


__all__ = ["TRACKED_SPACE", "maybe_uninit_locals"]
