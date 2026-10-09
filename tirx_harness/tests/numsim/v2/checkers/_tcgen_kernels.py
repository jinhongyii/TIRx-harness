"""Pure-TVM copies of the legacy tcgen05 TI16 / sparse-B16 / LUT-B kernel builders.

Copied verbatim from ``tests/numsim/runtime/test_tcgen05_ti16.py``
(``ti16_kernel``, ``sparse_b16_case``), ``tests/numsim/runtime/test_tcgen05_sparse_b16.py``
(``SPARSE_B16_CASES``, ``encode_b16``, ``sparse_float_case``) and
``tests/numsim/runtime/test_tcgen_lut_b.py`` (``lut_b_case``) and
``tests/numsim/runtime/test_tcgen05_i8.py`` (``i8_case``). Those modules
import the legacy NumSim through ``tests.numsim.support.execution``, so the v2
checker tests cannot import them.
"""

import numpy as np
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TileLayout


def ti16_kernel(
    enable_d,
    tmem_a=False,
    cta_group=1,
    m=128,
    ws=False,
    mask_form=True,
    kind="ti16",
    sparse=False,
    a_format=0,
    b_format=0,
    arch="sm_100a",
    saturate=False,
    early_reuse=None,
    restricted_multicast=False,
    half_accumulator=False,
    mma_k=None,
    transpose_a=False,
    transpose_b=False,
    collectors="",
    implicit_scale=False,
    lut_b=False,
    lut_segment=0,
    lut_address_offset=96,
    sparsity_selector=1,
):
    block_scale = kind in {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    wide_k = mma_k in (96, 128) if kind in {"mxf4", "mxf4nvf4"} else mma_k == 64
    extended_scales = wide_k and kind in {"mxf4", "mxf4nvf4"}
    n = 64 if ws else 16 * cta_group
    rows = m // cta_group
    half_layout = rows == 64 and not ws and (cta_group == 1 or sparse)
    columns = n if half_layout else n // (128 // rows)
    b_rows = n + 32 if ws else n // cta_group
    k = 64 if sparse and wide_k else 32 if sparse or wide_k else 16
    a_columns = 32 if wide_k else 16
    a_swizzle = SwizzleMode.SWIZZLE_64B_ATOM if wide_k else SwizzleMode.SWIZZLE_32B_ATOM
    a_layout = mma_shared_layout("uint16", a_swizzle, (rows, a_columns))
    b_layout = mma_shared_layout(
        "uint16",
        SwizzleMode.SWIZZLE_128B_ATOM
        if sparse and wide_k
        else SwizzleMode.SWIZZLE_64B_ATOM
        if sparse or wide_k
        else SwizzleMode.SWIZZLE_32B_ATOM,
        (b_rows, k),
    )

    # PTX Table 67: 32 TF32 rows x 4 K values per 512B swizzle atom.
    # The uint16 storage keeps each TF32 word's two halves adjacent.
    def mn_layout(extent, half_columns):
        if kind in {"f8f6f4", "mxf8f6f4"}:
            return ComposeLayout(
                4,
                1,
                3,
                TileLayout(S[(extent // 32, 32, half_columns // 4, 8) : (256, 1, extent * 8, 32)]),
            )
        if kind == "f16":
            return ComposeLayout(
                3,
                1,
                3,
                TileLayout(S[(extent // 16, 16, half_columns // 8, 8) : (128, 1, extent * 8, 16)]),
            )
        return ComposeLayout(
            4,
            2,
            2,
            TileLayout(
                S[(extent // 32, 32, half_columns // 8, 4, 2) : (256, 2, extent * 8, 64, 1)]
            ),
        )

    shared_a_rows, shared_b_rows = rows, b_rows
    if transpose_a or transpose_b:
        assert kind in {"tf32", "f16", "f8f6f4", "mxf8f6f4"}
    if transpose_a:
        a_layout = mn_layout(rows, a_columns)
    if transpose_b:
        atom_rows = 32 if kind in {"tf32", "f8f6f4", "mxf8f6f4"} else 16
        shared_b_rows = (b_rows + atom_rows - 1) // atom_rows * atom_rows
        b_layout = mn_layout(shared_b_rows, k)
    byte_a = kind in {"f8f6f4", "mxf8f6f4"} and transpose_a
    byte_b = kind in {"f8f6f4", "mxf8f6f4"} and transpose_b
    formats = (2 << 4) | (3 << 7) | (3 << 10) if kind == "ti16" else 1 << 4
    if kind == "tf32":
        formats = (1 << 4) | (2 << 7) | (2 << 10)
    if kind == "i8":
        formats = (2 << 4) | (a_format << 7) | (b_format << 10) | (int(saturate) << 3)
    if kind in {"f16", "f8f6f4"}:
        formats = (1 << 4) | (a_format << 7) | (b_format << 10)
        if half_accumulator:
            formats &= ~(1 << 4)
    if block_scale:
        formats = (1 << 23) | (1 << 29) | (2 << 4) | (a_format << 7) | (b_format << 10)
        if kind != "mxf8f6f4":
            formats = (1 << 23) | (1 << 7) | (1 << 10)
            if arch == "sm_107a":
                formats |= 1 << 12
    descriptor = formats | ((n // 8) << 17) | ((m // 16) << 24)
    descriptor |= (int(transpose_a) << 15) | (int(transpose_b) << 16)
    if wide_k:
        descriptor |= 1 << (
            (31 if mma_k == 96 else 3)
            if kind in {"mxf4", "mxf4nvf4"}
            else 31
            if block_scale
            else 29
        )
    if sparse:
        descriptor |= 4 | sparsity_selector
    if ws:
        descriptor |= 1 << 30  # Reserve capacity for a B-column shift of up to 8.
    masks = [0] * (4 * cta_group)
    if cta_group == 2:
        masks[4] = 2  # Paired CTA, physical lane 1; its destination must remain unchanged.
    mma = f"tcgen05.mma{'.ws' if ws else ''}{'.sp' if sparse else ''}.cta_group::{cta_group}.kind::{kind}"
    if block_scale:
        mma += ".block_scale" + ("" if implicit_scale else ".block32")
    mma += collectors
    if lut_b:
        assert kind in {"f8f6f4", "mxf8f6f4"} and mma_k == 64
        mma += ".decompress::lut::b"
    sparse_words = 4 if sparse and wide_k else 2
    metadata_columns = (
        2 + sparse_words if sparse and block_scale else 4 if lut_b and block_scale else sparse_words
    )
    lookup_column = 112 if (lut_b or sparse) and block_scale else 96
    lookup_word = 2 if (lut_b or sparse) and block_scale else 0
    alloc = f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"
    dealloc = f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"
    relinquish = f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"
    commit = f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
    restricted_commit = f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster{'.multicast::cluster::32b' if restricted_multicast else ''}.b64"
    st = f"tcgen05.st.sync.aligned.32x32b.x{columns}.b32"
    ld = f"tcgen05.ld.sync.aligned.32x32b.x{columns}.b32"
    st_a = f"tcgen05.st.sync.aligned.32x32b.x{a_columns // 2}.b32"
    scale_columns = 16 if extended_scales else 8
    st_scales = f"tcgen05.st.sync.aligned.32x32b.x{scale_columns}.b32"
    sfb_column = 104 if extended_scales else 100

    @T.prim_func
    def kernel(
        a: T.Buffer((4, m, a_columns), "uint16"),
        b: T.Buffer((b_rows * cta_group, k), "uint16"),
        zero_mask: T.Buffer((1,), "uint64"),
        metadata: T.Buffer((2, 128, metadata_columns), "uint32"),
        seed: T.Buffer((m, n), "int32"),
        out: T.Buffer((m, n), "int32"),
    ):
        T.device_entry()
        _cluster = T.cluster_id([1])
        cta = T.cta_id_in_cluster([cta_group])
        _wg = T.warpgroup_id([1])
        warp = T.warp_id_in_wg([4])
        lane = T.lane_id([32])
        address = T.alloc_buffer((1,), "uint32", scope="shared")
        barrier = T.alloc_buffer((3,), "uint64", scope="shared")
        # Relative swizzle layouts need a base aligned to their full 256B period.
        shared_a = T.alloc_buffer(
            (shared_a_rows, a_columns * (2 if byte_a else 1)),
            "uint8" if byte_a else "uint16",
            scope="shared",
            layout=a_layout,
            align=512,
        )
        shared_b = T.alloc_buffer(
            (shared_b_rows, k * (2 if byte_b else 1)),
            "uint8" if byte_b else "uint16",
            scope="shared",
            layout=b_layout,
            align=1024 if sparse and wide_k else 512,
        )
        regs = T.alloc_local((columns,), "int32")
        packed_a = T.alloc_local((a_columns // 2,), "uint32")
        desc_a: T.uint64
        desc_b: T.uint64
        row = cta * rows + (warp * 16 + lane % 16 if half_layout else (warp * 32 + lane) % rows)
        bank = 0 if half_layout else (warp * 32 + lane) // rows
        if warp == 0 and lane == 0:
            if byte_a:
                Tx.copy(shared_a[:rows, :], a.view("uint8")[0, cta * rows : (cta + 1) * rows, :])
            else:
                Tx.copy(shared_a[:rows, :], a[0, cta * rows : (cta + 1) * rows, :])
            if byte_b:
                Tx.copy(shared_b[:b_rows, :], b.view("uint8")[cta * b_rows : (cta + 1) * b_rows, :])
            else:
                Tx.copy(shared_b[:b_rows, :], b[cta * b_rows : (cta + 1) * b_rows, :])
        if warp == 0:
            T.ptx[alloc](address.ptr_to([0]), 128)
            if lane == 0:
                T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
                if early_reuse is not None:
                    T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([1]), 1)
                    T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([2]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        for i in T.unroll(columns):
            regs[i] = seed[row, bank * columns + i]
        T.ptx[st](address[0], *[regs[i] for i in range(columns)])
        if sparse or lut_b:
            T.ptx[f"tcgen05.st.sync.aligned.32x32b.x{sparse_words}.b32"](
                address[0] + T.uint32(lookup_column),
                *[metadata[cta, warp * 32 + lane, lookup_word + i] for i in range(sparse_words)],
            )
        if block_scale:
            T.ptx[st_scales](
                address[0] + T.uint32(96),
                *[
                    metadata[cta, warp * 32 + lane, i // (scale_columns // 2)]
                    for i in range(scale_columns)
                ],
            )
        if tmem_a:
            for i in T.unroll(a_columns // 2):
                packed_a[i] = T.Cast("uint32", a[bank, row, i * 2]) | (
                    T.Cast("uint32", a[bank, row, i * 2 + 1]) << 16
                )
            T.ptx[st_a](address[0] + T.uint32(64), *[packed_a[i] for i in range(a_columns // 2)])
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cluster_sync()
        if cta == 0 and warp == 0 and lane == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a),
                shared_a.ptr_to([0, 0]),
                ldo=(32 if kind == "tf32" else 16) if transpose_a else 0,
                sdo=rows // (2 if byte_a else 1) if transpose_a else 32 if wide_k else 16,
                swizzle=(4 if kind == "tf32" else 1) if transpose_a else 2 if wide_k else 1,
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b),
                shared_b.ptr_to([0, 0]),
                ldo=(32 if kind == "tf32" else 16) if transpose_b else 0,
                sdo=shared_b_rows // (2 if byte_b else 1)
                if transpose_b
                else 64
                if wide_k and sparse
                else 32
                if sparse or wide_k
                else 16,
                swizzle=(4 if kind == "tf32" else 1)
                if transpose_b
                else 3
                if wide_k and sparse
                else 2
                if sparse or wide_k
                else 1,
            )
            if lut_b:
                desc_b = desc_b | T.uint64(lut_segment << 53)
            T.ptx[mma](
                address[0],
                *([address[0] + T.uint32(64)] if tmem_a else [desc_a]),
                desc_b,
                *([address[0] + T.uint32(lookup_column)] if sparse else []),
                *([address[0] + T.uint32(lut_address_offset)] if lut_b else []),
                T.uint32(descriptor),
                *(
                    [address[0] + T.uint32(96), address[0] + T.uint32(sfb_column)]
                    if block_scale
                    else []
                    if ws
                    else [T.uint32(mask) for mask in masks]
                ),
                T.ptx.pred(T.uint32(enable_d)),
                *([zero_mask[0]] if ws and mask_form else []),
                pred=(zero_mask[0] >> 63) == T.uint64(0),
            )
            if early_reuse is not None:
                # A second commit has no pending A token, but must retain the
                # first commit's completed A frontier for its own mbarrier.
                for phase in T.unroll(2):
                    T.ptx[restricted_commit](
                        barrier.ptr_to([phase + 1]),
                        *([T.uint32((1 << cta_group) - 1)] if restricted_multicast else []),
                    )
                    T.cuda.mbarrier_wait(barrier.ptr_to([phase + 1]), 0)
                T.ptx.fence.proxy.async_.shared__cta()
                if early_reuse == "a":
                    shared_a[0, 0] = T.uint16(42)
                elif early_reuse == "b":
                    shared_b[0, 0] = T.uint16(42)
            if early_reuse not in ("lookup", "lookup_tail", "tmem_a_tail"):
                T.ptx[commit](barrier.ptr_to([0]))
        if early_reuse in ("lookup", "lookup_tail", "tmem_a_tail") and cta == 0 and warp == 0:
            # A shared-A restricted wait does not retire LUT/metadata or TMEM A.
            T.ptx["tcgen05.st.sync.aligned.32x32b.x2.b32"](
                address[0]
                + T.uint32(
                    72
                    if early_reuse == "tmem_a_tail"
                    else lookup_column + (2 if early_reuse == "lookup_tail" else 0)
                ),
                T.uint32(0),
                T.uint32(0),
            )
            T.ptx.tcgen05.wait__st.sync.aligned()
            if lane == 0:
                T.ptx[commit](barrier.ptr_to([0]))
        if cta == 0 and warp == 0:
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.cuda.cluster_sync()
        T.ptx[ld](*[regs[i] for i in range(columns)], address[0])
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for i in T.unroll(columns):
            if not half_layout or lane < 16:
                out[row, bank * columns + i] = regs[i]
        T.cuda.cluster_sync()
        if warp == 0:
            T.ptx[dealloc](address[0], 128)
            T.ptx[relinquish]()

    return kernel.with_attr("tirx.cuda_arch", "sm_107a" if kind == "ti16" else arch)


def sparse_b16_case(m, tmem_a, ws, *, mask_form=True, cta_group=1, collectors=""):
    n = 64 if ws else 16 * cta_group
    rows = m // cta_group
    banks = 128 // m if ws else 1
    a = np.stack(
        [np.arange(m * 16, dtype=np.int64).reshape(m, 16) % 11 + 1 + bank for bank in range(4)]
    )
    b = np.arange((n + 32 if ws else n) * 32, dtype=np.int64).reshape(-1, 32) % 7 + 1
    metadata = np.zeros((2, 128, 2), np.uint32)
    expanded = np.zeros((banks, m, 32), np.int64)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 1), (2, 3))
    for bank in range(banks):
        for row in range(m):
            local_row = row % rows
            physical_row = (
                local_row // 16 * 32 + local_row % 16
                if rows == 64 and not ws
                else bank * rows + local_row
            )
            for chunk in range(8):
                first, second = pairs[(bank + row + chunk) % len(pairs)]
                # PTX metadata figure: rows 0/8 share low/high nibbles, and
                # the second K half lives in lanes 8--15 of each 16-lane group.
                lane = (
                    physical_row // 32 * 32
                    + physical_row % 8
                    + physical_row % 32 // 16 * 16
                    + chunk // 4 * 8
                )
                shift = physical_row % 16 // 8 * 16 + chunk % 4 * 4
                metadata[row // rows, lane, 1] |= np.uint32((first | (second << 2)) << shift)
                values = a[bank if tmem_a else 0, row, chunk * 2 : chunk * 2 + 2]
                expanded[bank, row, chunk * 4 + first] = values[0]
                expanded[bank, row, chunk * 4 + second] = values[1]
    expected = np.concatenate(
        [
            expanded[bank] @ b[bank * (n // banks) : (bank + 1) * (n // banks)].T
            for bank in range(banks)
        ],
        axis=1,
    )
    if cta_group == 2:
        expected[rows + 1] = 0  # Second CTA's disabled physical lane 1.
    args = {
        "a": a.astype(np.uint16),
        "b": b.astype(np.uint16),
        "metadata": metadata,
        "zero_mask": np.zeros(1, np.uint64),
        "seed": np.zeros((m, n), np.int32),
        "out": np.zeros((m, n), np.int32),
    }
    return (
        ti16_kernel(
            False,
            tmem_a,
            cta_group=cta_group,
            m=m,
            ws=ws,
            mask_form=mask_form,
            sparse=True,
            collectors=collectors,
        ),
        args,
        expected,
    )


SPARSE_B16_CASES = [
    (1, 64, False, 0, 0, False, False, False),
    (1, 128, True, 1, 1, False, False, False),
    (2, 128, True, 0, 0, True, False, False),
    (2, 256, False, 0, 0, True, False, False),
    (1, 128, False, 1, 1, False, True, True),
    (2, 256, True, 0, 0, False, False, True),
]


def encode_b16(values, bf16):
    return (
        (values.view(np.uint32) >> 16).astype(np.uint16)
        if bf16
        else values.astype(np.float16).view(np.uint16)
    )


def sparse_float_case(cta, m, tmem, af, bf, half, ta, tb, *, selector=1, early_reuse=None):
    _, args, dot = sparse_b16_case(m, tmem, False, cta_group=cta)
    a = -args["a"].astype(np.float32) / 8
    b = args["b"].astype(np.float32) / 16
    args["a"], args["b"] = encode_b16(a, af), encode_b16(b, bf)
    seed = (np.arange(m * (16 * cta)).reshape(m, -1) % 7 / 16).astype(np.float32)
    expected = dot.astype(np.float32) * (-1 / 128) + seed
    args["seed"] = (
        (seed.astype(np.float16).view(np.uint16).astype(np.uint32) | np.uint32(0xBEEF0000)).view(
            np.int32
        )
        if half
        else seed.view(np.int32)
    )
    expected = (
        expected.astype(np.float16).view(np.uint16).astype(np.int32)
        if half
        else expected.view(np.int32)
    )
    if cta == 2:
        expected[m // 2 + 1] = args["seed"][m // 2 + 1]
    if selector == 0:
        args["metadata"][:, :, 0] = args["metadata"][:, :, 1]
        args["metadata"][:, :, 1] = 0
    kernel = ti16_kernel(
        True,
        tmem,
        cta_group=cta,
        m=m,
        kind="f16",
        sparse=True,
        a_format=af,
        b_format=bf,
        half_accumulator=half,
        transpose_a=ta,
        transpose_b=tb,
        sparsity_selector=selector,
        early_reuse=early_reuse,
        collectors=".collector::a::discard.collector::b::discard",
        arch="sm_107a",
    )
    return kernel, args, expected


def lut_b_case(group=1, tmem=False, segment=0, half=False, block=False, **options):
    m, n, k = 128 * group, 16 * group, 64
    kernel = ti16_kernel(
        True,
        tmem_a=tmem,
        cta_group=group,
        m=m,
        kind="mxf8f6f4" if block else "f8f6f4",
        a_format=3 if block and tmem else 0,
        arch="sm_107a",
        mma_k=k,
        half_accumulator=half,
        lut_b=True,
        lut_segment=segment,
        **({"lut_address_offset": 112} if block else {}),
        **options,
    )
    fp8_bits = np.array([0, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E], np.uint8)
    a = (np.arange(4 * m * k).reshape(4, m, k) % 7 - 3).astype(np.float32)
    a_bits = fp8_bits[np.abs(a).astype(np.int32)] | ((a < 0).astype(np.uint8) << 7)
    if block and tmem:
        # Table 66: K64 MXF8F6F4 TMEM A uses packed 6-bit containers.
        codes = np.array([0, 8, 16, 20], np.uint8)[np.abs(a).astype(np.int32)]
        codes |= (a < 0).astype(np.uint8) << 5
        a_bits = np.zeros_like(codes)
        for bank in range(4):
            for row in range(m):
                packed = sum(int(value) << (6 * i) for i, value in enumerate(codes[bank, row]))
                a_bits[bank, row, :48] = np.frombuffer(packed.to_bytes(48, "little"), np.uint8)
    b_bits = np.zeros((n, 64), np.uint8)
    lookup = np.zeros((2, 128, 8), np.uint8)
    b = np.empty((n, k), np.float32)
    for cta in range(group):
        for lut_row in range(2):
            values = np.roll(np.arange(8, dtype=np.float32), cta + lut_row)
            values *= -1 if (cta + lut_row) % 2 else 1
            lookup[cta, lut_row] = fp8_bits[np.abs(values).astype(np.int32)] | (
                (values < 0).astype(np.uint8) << 7
            )
            for row in range(lut_row * 8, lut_row * 8 + 8):
                global_row = cta * 16 + row
                indices = (np.arange(128) * 3 + row + cta) % 8
                indices[64:] = (indices[64:] + 5) % 8
                # Independent little-endian packing, including fields crossing bytes.
                packed = sum(int(index) << (3 * i) for i, index in enumerate(indices))
                b_bits[global_row, :48] = np.frombuffer(packed.to_bytes(48, "little"), np.uint8)
                b[global_row] = values[indices[segment * 64 : (segment + 1) * 64]]
    seed = np.full((m, n), 0.5, np.float32)
    metadata = lookup.view(np.uint32).reshape(2, 128, 2)
    if block:
        metadata = np.zeros((2, 128, 4), np.uint32)
        metadata[:, :, 2:] = lookup.view(np.uint32).reshape(2, 128, 2)
        lane = np.arange(128, dtype=np.uint32)
        for cta in range(group):
            metadata[cta, :, 0] = (126 + lane % 2 + cta) << 8  # SFA_ID = 1.
            metadata[cta, :, 1] = (127 + lane // 16 % 2) << 16  # SFB_ID = 2.
        a *= np.exp2((np.arange(m) % 2 - 1 + np.arange(m) // 128)[:, None])
        # CTA2 uses replicated, joint-N B scales, unlike the local-N lookup tables.
        b *= np.exp2((np.arange(n) // 16 % 2)[:, None])

    def encode_result(value):
        return (
            value.astype(np.float16).view(np.uint16).astype(np.int32)
            if half
            else value.view(np.int32)
        )

    expected = a[0] @ b.T + seed
    if group == 2 and not block:
        expected[129] = seed[129]
    args = {
        "a": a_bits.view(np.uint16),
        "b": b_bits.view(np.uint16),
        "zero_mask": np.zeros(1, np.uint64),
        "metadata": metadata,
        "seed": encode_result(seed),
        "out": np.zeros((m, n), np.int32),
    }
    return kernel, args, encode_result(expected)


# Copied verbatim from tests/numsim/runtime/test_tcgen05_i8.py (legacy-importing module).
def i8_case(cta_group, m, tmem_a, ws, a_format, b_format, saturate, **options):
    n = 64 if ws else 16 * cta_group
    rows = m // cta_group
    rng = np.random.default_rng(104)
    a_bits = rng.integers(0, 256, (4, m, 32), np.uint8)
    b_bits = rng.integers(0, 256, (n + 32 if ws else n, 32), np.uint8)
    a = a_bits.view(np.int8 if a_format else np.uint8).astype(np.int64)
    b = b_bits.view(np.int8 if b_format else np.uint8).astype(np.int64)
    seed = (
        np.broadcast_to(np.where(np.arange(n) % 2, -(2**31) + 64, 2**31 - 65), (m, n))
        .astype(np.int32)
        .copy()
    )
    banks = (128 // rows) if tmem_a and (ws or cta_group == 2) else 1
    expected = np.concatenate(
        [a[bank] @ b[bank * (n // banks) : (bank + 1) * (n // banks)].T for bank in range(banks)],
        axis=1,
    )
    expected += seed.astype(np.int64)
    if saturate:
        expected = expected.clip(-(2**31), 2**31 - 1)
    if cta_group == 2:
        kept = n // (128 // rows)
        expected[rows + 1, :kept] = seed[rows + 1, :kept]
    args = {
        "a": a_bits.view(np.uint16),
        "b": b_bits.view(np.uint16),
        "zero_mask": np.zeros(1, np.uint64),
        "metadata": np.zeros((2, 128, 2), np.uint32),
        "seed": seed,
        "out": np.zeros_like(seed),
    }
    kernel = ti16_kernel(
        True,
        tmem_a,
        cta_group,
        m,
        ws=ws,
        kind="i8",
        a_format=a_format,
        b_format=b_format,
        saturate=saturate,
        **options,
    )
    return kernel, args, expected.astype(np.int32)
