"""TIRx function(s) -> serialized ``numsim_core::Module`` with a content-addressed cache.

``transpile(func)`` lowers each PrimFunc with ``v2.lowering`` (owned by W1),
wraps the programs in a ``Module`` (``{"format_version", "kernels"}``),
validates it in Rust (``numsim_core_py.load_module``) and caches the module
bytes under ``$NUMSIM_CACHE_DIR/v2-modules/<sha256>.module.json``.

The cache key hashes the TIRx source (``tvm.ir.save_json``), the module
format version and the lowering package's own source, so a lowering change
never serves a stale module.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

from tirx_harness.numsim.errors import NumSimError, UnsupportedTIRxError

from .options import options


class ModuleContractError(NumSimError, ValueError):
    """Lowering produced a module ``numsim-core`` rejects (decode/validate)."""


def native():
    """The ``numsim_core_py`` extension (see ``core-rs/numsim-py/build_dev.sh``)."""

    try:
        from . import numsim_core_py  # type: ignore[attr-defined]
    except ImportError as error:
        raise NotImplementedError(
            "numsim_core_py is not built; run "
            "tirx_harness/src/tirx_harness/numsim/core-rs/numsim-py/build_dev.sh"
        ) from error
    return numsim_core_py


# --------------------------------------------------------------------------
# Static facts read from the module JSON (no engine types)


def _span_dict(span: dict[str, Any]) -> dict[str, Any] | None:
    """``site::Span`` -> the serialized ``SourceSpan`` shape used in payloads."""

    if not span or not span.get("line"):
        return None
    return {
        "kind": "span",
        "source_name": span.get("file") or "<unknown>",
        "line": int(span["line"]),
        "column": int(span.get("col", 0)),
        "end_line": int(span.get("end_line", span["line"])),
        "end_column": int(span.get("end_col", 0)),
    }


def site_source_span(site: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialized source span of one ``SiteInfo`` (outermost call first)."""

    if not site:
        return None
    spans = [s for s in (_span_dict(span) for span in site.get("spans", ())) if s is not None]
    if not spans:
        return None
    if len(spans) == 1:
        return spans[0]
    # SiteInfo.spans is innermost first; SequentialSourceSpan is outermost first.
    return {"kind": "sequential", "spans": list(reversed(spans))}


@dataclass(frozen=True)
class KernelSpec:
    """Per-kernel facts the Python layer needs (names, ABI, sites)."""

    index: int
    name: str
    host_abi: tuple[dict[str, Any], ...]
    sites: tuple[dict[str, Any], ...]
    # Legacy attribute read by conformance tooling; v2 embeds spans instead.
    source_map: tuple = ()
    # Launch topology (grid/cluster/block; grid dims are `{"Const": n}` or
    # host expressions), for mapping legacy `cta_ids` subsets to clusters.
    topology: dict[str, Any] = field(default_factory=dict, compare=False)

    def source_span(self, site: int | None) -> dict[str, Any] | None:
        if site is None or site < 0 or site >= len(self.sites):
            return None
        return site_source_span(self.sites[site])


@dataclass(frozen=True)
class ModuleSpec:
    kernels: tuple[KernelSpec, ...]


def _module_spec(module: dict[str, Any]) -> ModuleSpec:
    return ModuleSpec(
        tuple(
            KernelSpec(
                index=index,
                name=str(kernel.get("name", f"kernel{index}")),
                host_abi=tuple(kernel.get("host_abi", ())),
                sites=tuple(kernel.get("sites", ())),
                topology=dict(kernel.get("topology") or {}),
            )
            for index, kernel in enumerate(module.get("kernels", ()))
        )
    )


@dataclass(frozen=True)
class CompiledModule:
    """A validated module: JSON bytes, static spec and the native handle."""

    data: bytes
    cache_key: str
    cache_path: Path | None = field(default=None, compare=False)
    # Milliseconds spent producing this module (lowering + validation, or
    # the cache load) and whether it came from the module cache.
    lower_ms: float = field(default=0.0, compare=False)
    cache_hit: bool = field(default=False, compare=False)

    @cached_property
    def document(self) -> dict[str, Any]:
        return json.loads(self.data)

    @cached_property
    def spec(self) -> ModuleSpec:
        return _module_spec(self.document)

    @cached_property
    def handle(self) -> Any:
        try:
            return native().load_module(self.data)
        except ValueError as error:
            raise ModuleContractError(str(error)) from error

    def prefix(self, count: int) -> CompiledModule:
        """The module of the first ``count`` kernels (phase-prefix runs)."""

        kernels = self.document["kernels"]
        if count == len(kernels):
            return self
        if not 1 <= count <= len(kernels):
            raise ValueError(f"prefix of {count} kernels outside [1, {len(kernels)}]")
        document = {**self.document, "kernels": kernels[:count]}
        data = json.dumps(document, separators=(",", ":")).encode()
        return CompiledModule(data=data, cache_key=f"{self.cache_key}:prefix{count}")


# --------------------------------------------------------------------------
# Lowering + cache


@lru_cache(maxsize=1)
def lowering_fingerprint() -> str:
    """Hash of the lowering package's sources (cache invalidation)."""

    root = Path(__file__).resolve().parent / "lowering"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def format_version() -> int:
    try:
        return int(native().program_format_version())
    except NotImplementedError:
        return 1


def _functions(func: Any) -> tuple[Any, ...]:
    return tuple(func) if isinstance(func, (list, tuple)) else (func,)


def _source_text(func: Any) -> str:
    import tvm

    try:
        return tvm.ir.save_json(func)
    except Exception:
        return func.script(show_meta=True)


def source_key(funcs: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(f"numsim-v2-module\0{format_version()}\0{lowering_fingerprint()}\0".encode())
    for func in funcs:
        digest.update(_source_text(func).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def lower_module(funcs: tuple[Any, ...]) -> dict[str, Any]:
    """Lower each PrimFunc and wrap the programs in a Module document."""

    from . import lowering

    lower_many = getattr(lowering, "lower_module", None)
    if lower_many is not None:
        module = lower_many(funcs)
        if isinstance(module, dict):
            return module
        if hasattr(module, "to_dict"):
            return module.to_dict()
        return json.loads(module)
    kernels = [lowering.lower(func).to_dict() for func in funcs]
    return {"format_version": format_version(), "kernels": kernels}


def from_document(
    document: dict[str, Any] | str | bytes, *, cache_key: str | None = None
) -> CompiledModule:
    """Wrap an already-lowered Module document (tests, fixtures)."""

    if isinstance(document, dict):
        data = json.dumps(document, separators=(",", ":")).encode()
    else:
        data = document.encode() if isinstance(document, str) else bytes(document)
    key = cache_key or hashlib.sha256(data).hexdigest()
    module = CompiledModule(data=data, cache_key=key)
    module.handle  # validate eagerly
    return module


def transpile(
    func: Any,
    *,
    cache_dir: str | Path | None = None,
    _default_generated_opt_level: int = 3,
    _analysis_capable: bool = False,
    _analysis_checker: str | None = None,
) -> CompiledModule:
    """Lower ``func`` (one PrimFunc or a sequence of launches) to a module.

    The underscore options exist for signature compatibility with the legacy
    ``numsim.transpile``: v2 runs every mode from the same module.
    """

    del _default_generated_opt_level, _analysis_capable
    if _analysis_checker not in {None, "synccheck", "racecheck"}:
        raise ValueError(f"unknown native analysis checker: {_analysis_checker!r}")
    funcs = _functions(func)
    opts = options()
    key = source_key(funcs)
    cache = Path(cache_dir) / "v2-modules" if cache_dir is not None else opts.module_cache_dir
    path = cache / f"{key}.module.json"
    started = time.perf_counter()
    if opts.use_cache and path.exists():
        try:
            cached = from_document(path.read_bytes(), cache_key=key)
            return replace(
                cached,
                cache_path=path,
                lower_ms=(time.perf_counter() - started) * 1e3,
                cache_hit=True,
            )
        except ModuleContractError:
            # The contract changed without a format-version bump: the cached
            # module no longer decodes. Re-lower instead of failing.
            path.unlink(missing_ok=True)

    from .lowering import LoweringUnsupported

    try:
        document = lower_module(funcs)
    except LoweringUnsupported as error:
        reasons = tuple(getattr(error, "reasons", ()) or ())
        raise UnsupportedTIRxError(str(error), unsupported=reasons) from error
    module = from_document(document, cache_key=key)
    if opts.use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(module.data)
        tmp.replace(path)
    return CompiledModule(
        data=module.data,
        cache_key=key,
        cache_path=path if opts.use_cache else None,
        lower_ms=(time.perf_counter() - started) * 1e3,
    )


__all__ = [
    "CompiledModule",
    "KernelSpec",
    "ModuleContractError",
    "ModuleSpec",
    "from_document",
    "lower_module",
    "native",
    "site_source_span",
    "transpile",
]
