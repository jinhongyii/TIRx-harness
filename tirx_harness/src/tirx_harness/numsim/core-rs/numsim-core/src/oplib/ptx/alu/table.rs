//! Modifier-slot table of the ALU ops (generated from the TIRx PTX op
//! table by `gen_table.py`; `slot?` = optional). Order = canonical
//! token order of `OpKey.mods`.

/// `(op, [(slot, optional, choices)])`.
pub(super) type SlotSpec = (&'static str, bool, &'static [&'static str]);

pub(super) const TABLE: &[(&str, &[SlotSpec])] = &[
    ("tirx.ptx.abs", &[("type", false, &["s16", "s32", "s64"])]),
    (
        "tirx.ptx.abs_f",
        &[("ftz", true, &["ftz"]), ("type", false, &["f32", "f64"])],
    ),
    (
        "tirx.ptx.abs_half",
        &[
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.add",
        &[
            ("rnd", true, &["rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f32", "f64", "f32x2"]),
            ("srctype", true, &["f16", "bf16"]),
        ],
    ),
    (
        "tirx.ptx.add_half",
        &[
            ("rnd", true, &["rn"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.add_int",
        &[
            ("sat", true, &["sat"]),
            (
                "type",
                false,
                &["u16", "u32", "u64", "u16x2", "s16", "s32", "s64", "s16x2"],
            ),
        ],
    ),
    (
        "tirx.ptx.add_mixed_vec_down_bf16",
        &[
            ("rnd", false, &["rz"]),
            ("dtype", false, &["bf16x2"]),
            ("atype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.add_mixed_vec_down_f16",
        &[
            ("rnd", false, &["rz"]),
            ("ftz", false, &["ftz"]),
            ("dtype", false, &["f16x2"]),
            ("atype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.add_mixed_vec_up",
        &[
            ("rnd", true, &["rn", "rz", "rm", "rp"]),
            ("dtype", false, &["f32x2"]),
            ("atype", false, &["f16x2", "bf16x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.and",
        &[("type", false, &["pred", "b16", "b32", "b64"])],
    ),
    (
        "tirx.ptx.bfe",
        &[("type", false, &["u32", "u64", "s32", "s64"])],
    ),
    ("tirx.ptx.bfi", &[("type", false, &["b32", "b64"])]),
    (
        "tirx.ptx.bfind",
        &[
            ("shiftamt", true, &["shiftamt"]),
            ("type", false, &["u32", "u64", "s32", "s64"]),
        ],
    ),
    (
        "tirx.ptx.bmsk",
        &[
            ("mode", false, &["clamp", "wrap"]),
            ("type", false, &["b32"]),
        ],
    ),
    ("tirx.ptx.brev", &[("type", false, &["b32", "b64"])]),
    (
        "tirx.ptx.clmad",
        &[("mode", false, &["hi", "lo"]), ("type", false, &["u64"])],
    ),
    ("tirx.ptx.clz", &[("type", false, &["b32", "b64"])]),
    ("tirx.ptx.cnot", &[("type", false, &["b16", "b32", "b64"])]),
    ("tirx.ptx.copysign", &[("type", false, &["f32", "f64"])]),
    (
        "tirx.ptx.cos",
        &[
            ("mode", false, &["approx"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32"]),
        ],
    ),
    (
        "tirx.ptx.createpolicy_cvt",
        &[
            ("kind", false, &["cvt"]),
            ("level", false, &["L2"]),
            ("type", false, &["b64"]),
        ],
    ),
    (
        "tirx.ptx.createpolicy_fraction",
        &[
            ("kind", false, &["fractional"]),
            (
                "pri",
                false,
                &[
                    "L2::evict_last",
                    "L2::evict_normal",
                    "L2::evict_first",
                    "L2::evict_unchanged",
                ],
            ),
            ("sec", true, &["L2::evict_first", "L2::evict_unchanged"]),
            ("type", false, &["b64"]),
        ],
    ),
    (
        "tirx.ptx.createpolicy_fractional",
        &[
            ("kind", false, &["fractional"]),
            (
                "pri",
                false,
                &[
                    "L2::evict_last",
                    "L2::evict_normal",
                    "L2::evict_first",
                    "L2::evict_unchanged",
                ],
            ),
            ("sec", true, &["L2::evict_first", "L2::evict_unchanged"]),
            ("type", false, &["b64"]),
        ],
    ),
    (
        "tirx.ptx.createpolicy_range",
        &[
            ("kind", false, &["range"]),
            ("space", true, &["global"]),
            (
                "pri",
                false,
                &[
                    "L2::evict_last",
                    "L2::evict_normal",
                    "L2::evict_first",
                    "L2::evict_unchanged",
                ],
            ),
            ("sec", true, &["L2::evict_first", "L2::evict_unchanged"]),
            ("type", false, &["b64"]),
        ],
    ),
    (
        "tirx.ptx.div",
        &[("type", false, &["u16", "u32", "u64", "s16", "s32", "s64"])],
    ),
    (
        "tirx.ptx.div_f",
        &[
            ("mode", false, &["approx", "full", "rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32", "f64"]),
        ],
    ),
    (
        "tirx.ptx.dp2a",
        &[
            ("mode", false, &["lo", "hi"]),
            ("atype", false, &["u32", "s32"]),
            ("btype", false, &["u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.dp4a",
        &[
            ("atype", false, &["u32", "s32"]),
            ("btype", false, &["u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.ex2",
        &[
            ("mode", false, &["approx"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32"]),
        ],
    ),
    (
        "tirx.ptx.ex2_half",
        &[
            ("mode", false, &["approx"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.fma",
        &[
            ("rnd", false, &["rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f32", "f64", "f32x2"]),
            ("srctype", true, &["f16", "bf16"]),
        ],
    ),
    (
        "tirx.ptx.fma_half",
        &[
            ("rnd", false, &["rn"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("oob", true, &["oob"]),
            ("relu", true, &["relu"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.fma_mixed_vec",
        &[
            ("rnd", false, &["rn", "rz", "rm", "rp"]),
            ("dtype", false, &["f32x2"]),
            ("atype", false, &["f16x2", "bf16x2"]),
            ("btype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    ("tirx.ptx.fns", &[("type", false, &["b32"])]),
    (
        "tirx.ptx.lg2",
        &[
            ("mode", false, &["approx"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32"]),
        ],
    ),
    ("tirx.ptx.lop3", &[("type", false, &["b32"])]),
    (
        "tirx.ptx.lop3_bool",
        &[("boolop", false, &["or", "and"]), ("type", false, &["b32"])],
    ),
    (
        "tirx.ptx.lop3_bool_sink",
        &[("boolop", false, &["or", "and"]), ("type", false, &["b32"])],
    ),
    (
        "tirx.ptx.mad24",
        &[
            ("mode", false, &["hi", "lo"]),
            ("sat", true, &["sat"]),
            ("type", false, &["u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.mad_f",
        &[
            ("rnd", false, &["rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f32", "f64"]),
        ],
    ),
    (
        "tirx.ptx.mad_int",
        &[
            ("mode", false, &["hi", "lo"]),
            ("sat", true, &["sat"]),
            ("type", false, &["u16", "u32", "u64", "s16", "s32", "s64"]),
        ],
    ),
    (
        "tirx.ptx.mad_wide",
        &[
            ("mode", false, &["wide"]),
            ("type", false, &["u16", "s16", "u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.max",
        &[
            ("ftz", true, &["ftz"]),
            ("nan", true, &["NaN"]),
            ("xorsign", true, &["xorsign"]),
            ("abs", true, &["abs"]),
            ("relu", true, &["relu"]),
            (
                "type",
                false,
                &[
                    "f32", "f64", "f16", "f16x2", "bf16", "bf16x2", "u16", "u32", "u64", "u16x2",
                    "s16", "s64", "s16x2", "s32",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.max3",
        &[
            ("ftz", true, &["ftz"]),
            ("nan", true, &["NaN"]),
            ("abs", true, &["abs"]),
            ("type", false, &["f32"]),
        ],
    ),
    (
        "tirx.ptx.min",
        &[
            ("ftz", true, &["ftz"]),
            ("nan", true, &["NaN"]),
            ("xorsign", true, &["xorsign"]),
            ("abs", true, &["abs"]),
            ("relu", true, &["relu"]),
            (
                "type",
                false,
                &[
                    "f32", "f64", "f16", "f16x2", "bf16", "bf16x2", "u16", "u32", "u64", "u16x2",
                    "s16", "s64", "s16x2", "s32",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.min3",
        &[
            ("ftz", true, &["ftz"]),
            ("nan", true, &["NaN"]),
            ("abs", true, &["abs"]),
            ("type", false, &["f32"]),
        ],
    ),
    (
        "tirx.ptx.mov",
        &[(
            "type",
            false,
            &[
                "pred", "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
            ],
        )],
    ),
    ("tirx.ptx.mov_pack_b16x2", &[("type", false, &["b32"])]),
    ("tirx.ptx.mov_pack_b16x4", &[("type", false, &["b64"])]),
    ("tirx.ptx.mov_pack_b32x2", &[("type", false, &["b64"])]),
    ("tirx.ptx.mov_pack_b32x4", &[("type", false, &["b128"])]),
    ("tirx.ptx.mov_pack_b64x2", &[("type", false, &["b128"])]),
    ("tirx.ptx.mov_unpack_b16x2", &[("type", false, &["b32"])]),
    ("tirx.ptx.mov_unpack_b16x4", &[("type", false, &["b64"])]),
    ("tirx.ptx.mov_unpack_b32x2", &[("type", false, &["b64"])]),
    ("tirx.ptx.mov_unpack_b32x4", &[("type", false, &["b128"])]),
    ("tirx.ptx.mov_unpack_b64x2", &[("type", false, &["b128"])]),
    (
        "tirx.ptx.mul",
        &[
            ("rnd", true, &["rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f32", "f64", "f32x2"]),
        ],
    ),
    (
        "tirx.ptx.mul24",
        &[
            ("mode", false, &["hi", "lo"]),
            ("type", false, &["u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.mul_half",
        &[
            ("rnd", true, &["rn"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.mul_int",
        &[
            ("mode", false, &["hi", "lo"]),
            ("type", false, &["u16", "u32", "u64", "s16", "s32", "s64"]),
        ],
    ),
    (
        "tirx.ptx.mul_mixed_vec_bf16_f16",
        &[
            ("dtype", false, &["bf16x2"]),
            ("atype", false, &["bf16x2"]),
            ("ctype", false, &["f16x2"]),
        ],
    ),
    (
        "tirx.ptx.mul_mixed_vec_down_bf16",
        &[
            ("rnd", false, &["rz"]),
            ("dtype", false, &["bf16x2"]),
            ("atype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.mul_mixed_vec_down_f16",
        &[
            ("ftz", false, &["ftz"]),
            ("rnd", false, &["rz"]),
            ("dtype", false, &["f16x2"]),
            ("atype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.mul_mixed_vec_f16_bf16",
        &[
            ("dtype", false, &["f16x2"]),
            ("atype", false, &["f16x2"]),
            ("ctype", false, &["bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.mul_wide",
        &[
            ("mode", false, &["wide"]),
            ("type", false, &["u16", "s16", "u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.neg",
        &[("ftz", true, &["ftz"]), ("type", false, &["f32", "f64"])],
    ),
    (
        "tirx.ptx.neg_half",
        &[
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.neg_int",
        &[("type", false, &["s16", "s32", "s64"])],
    ),
    (
        "tirx.ptx.not",
        &[("type", false, &["pred", "b16", "b32", "b64"])],
    ),
    (
        "tirx.ptx.or",
        &[("type", false, &["pred", "b16", "b32", "b64"])],
    ),
    ("tirx.ptx.popc", &[("type", false, &["b32", "b64"])]),
    (
        "tirx.ptx.prmt",
        &[
            ("type", false, &["b32"]),
            ("mode", true, &["f4e", "b4e", "rc8", "ecl", "ecr", "rc16"]),
        ],
    ),
    (
        "tirx.ptx.rcp",
        &[
            ("mode", false, &["approx", "rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32", "f64"]),
        ],
    ),
    (
        "tirx.ptx.rem",
        &[("type", false, &["u16", "u32", "u64", "s16", "s32", "s64"])],
    ),
    (
        "tirx.ptx.rsqrt",
        &[
            ("mode", false, &["approx"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32", "f64"]),
        ],
    ),
    (
        "tirx.ptx.sad",
        &[("type", false, &["u16", "u32", "u64", "s16", "s32", "s64"])],
    ),
    (
        "tirx.ptx.selp",
        &[(
            "type",
            false,
            &[
                "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
            ],
        )],
    ),
    (
        "tirx.ptx.set",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
                    "ltu", "leu", "gtu", "geu", "num", "nan",
                ],
            ),
            ("ftz", true, &["ftz"]),
            ("dtype", false, &["u32", "s32", "f32"]),
            (
                "stype",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.set_bool",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
                    "ltu", "leu", "gtu", "geu", "num", "nan",
                ],
            ),
            ("boolop", false, &["and", "or", "xor"]),
            ("ftz", true, &["ftz"]),
            ("dtype", false, &["u32", "s32", "f32"]),
            (
                "stype",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.set_half",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                    "num", "nan",
                ],
            ),
            ("ftz", true, &["ftz"]),
            (
                "dtype",
                false,
                &["f16", "bf16", "u16", "s16", "u32", "s32", "f16x2", "bf16x2"],
            ),
            (
                "stype",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f16", "f32",
                    "f64", "bf16", "f16x2", "bf16x2",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.set_half_bool",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                    "num", "nan",
                ],
            ),
            ("boolop", false, &["and", "or", "xor"]),
            ("ftz", true, &["ftz"]),
            (
                "dtype",
                false,
                &["f16", "bf16", "u16", "s16", "u32", "s32", "f16x2", "bf16x2"],
            ),
            (
                "stype",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f16", "f32",
                    "f64", "bf16", "f16x2", "bf16x2",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.set_packed",
        &[
            (
                "cmp",
                false,
                &["eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs"],
            ),
            ("type", false, &["u8x4", "s8x4", "u16x2", "s16x2"]),
        ],
    ),
    (
        "tirx.ptx.setp",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
                    "ltu", "leu", "gtu", "geu", "num", "nan",
                ],
            ),
            ("ftz", true, &["ftz"]),
            (
                "type",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.setp_bool",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
                    "ltu", "leu", "gtu", "geu", "num", "nan",
                ],
            ),
            ("boolop", false, &["and", "or", "xor"]),
            ("ftz", true, &["ftz"]),
            (
                "type",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.setp_bool_pq",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
                    "ltu", "leu", "gtu", "geu", "num", "nan",
                ],
            ),
            ("boolop", false, &["and", "or", "xor"]),
            ("ftz", true, &["ftz"]),
            (
                "type",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.setp_half",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                    "num", "nan",
                ],
            ),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16", "bf16"]),
        ],
    ),
    (
        "tirx.ptx.setp_half_bool",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                    "num", "nan",
                ],
            ),
            ("boolop", false, &["and", "or", "xor"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16", "bf16"]),
        ],
    ),
    (
        "tirx.ptx.setp_half_bool_pq",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                    "num", "nan",
                ],
            ),
            ("boolop", false, &["and", "or", "xor"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16x2", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.setp_half_pq",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                    "num", "nan",
                ],
            ),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f16x2", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.setp_pq",
        &[
            (
                "cmp",
                false,
                &[
                    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
                    "ltu", "leu", "gtu", "geu", "num", "nan",
                ],
            ),
            ("ftz", true, &["ftz"]),
            (
                "type",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
        ],
    ),
    (
        "tirx.ptx.shf",
        &[
            ("dir", false, &["l", "r"]),
            ("mode", false, &["clamp", "wrap"]),
            ("type", false, &["b32"]),
        ],
    ),
    ("tirx.ptx.shl", &[("type", false, &["b16", "b32", "b64"])]),
    (
        "tirx.ptx.shr",
        &[(
            "type",
            false,
            &[
                "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64",
            ],
        )],
    ),
    (
        "tirx.ptx.sin",
        &[
            ("mode", false, &["approx"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32"]),
        ],
    ),
    (
        "tirx.ptx.slct",
        &[
            ("ftz", true, &["ftz"]),
            (
                "dtype",
                false,
                &[
                    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
                ],
            ),
            ("ctype", false, &["s32", "f32"]),
        ],
    ),
    (
        "tirx.ptx.spcompress",
        &[
            ("elemsize", false, &["b8", "b16"]),
            ("idxsize", false, &["b2", "b4"]),
            ("spfactor", false, &["sp::2:4"]),
            ("num", false, &["x1", "x2", "x4", "x8", "x16", "x32", "x64"]),
        ],
    ),
    (
        "tirx.ptx.spdecompress",
        &[
            ("elemsize", false, &["b8", "b16"]),
            ("idxsize", false, &["b2", "b4"]),
            (
                "spfactor",
                false,
                &[
                    "sp::1:2", "sp::1:4", "sp::1:8", "sp::1:16", "sp::2:4", "sp::2:8", "sp::2:16",
                    "sp::4:8", "sp::4:16",
                ],
            ),
            ("num", false, &["x1", "x2", "x4", "x8", "x16", "x32", "x64"]),
        ],
    ),
    (
        "tirx.ptx.sqrt",
        &[
            ("mode", false, &["approx", "rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("type", false, &["f32", "f64"]),
        ],
    ),
    (
        "tirx.ptx.sub",
        &[
            ("rnd", true, &["rn", "rz", "rm", "rp"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f32", "f64", "f32x2"]),
            ("srctype", true, &["f16", "bf16"]),
        ],
    ),
    (
        "tirx.ptx.sub_half",
        &[
            ("rnd", true, &["rn"]),
            ("ftz", true, &["ftz"]),
            ("sat", true, &["sat"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.sub_int",
        &[
            ("sat", true, &["sat"]),
            ("type", false, &["u16", "u32", "u64", "s16", "s32", "s64"]),
        ],
    ),
    (
        "tirx.ptx.sub_mixed_vec_down_bf16",
        &[
            ("rnd", false, &["rz"]),
            ("dtype", false, &["bf16x2"]),
            ("atype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.sub_mixed_vec_down_f16",
        &[
            ("rnd", false, &["rz"]),
            ("ftz", false, &["ftz"]),
            ("dtype", false, &["f16x2"]),
            ("atype", false, &["f32x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.sub_mixed_vec_up",
        &[
            ("rnd", true, &["rn", "rz", "rm", "rp"]),
            ("dtype", false, &["f32x2"]),
            ("atype", false, &["f16x2", "bf16x2"]),
            ("ctype", false, &["f32x2"]),
        ],
    ),
    (
        "tirx.ptx.szext",
        &[
            ("mode", false, &["clamp", "wrap"]),
            ("type", false, &["u32", "s32"]),
        ],
    ),
    (
        "tirx.ptx.tanh",
        &[("mode", false, &["approx"]), ("type", false, &["f32"])],
    ),
    (
        "tirx.ptx.tanh_half",
        &[
            ("mode", false, &["approx"]),
            ("type", false, &["f16", "f16x2", "bf16", "bf16x2"]),
        ],
    ),
    (
        "tirx.ptx.testp",
        &[
            (
                "op",
                false,
                &[
                    "finite",
                    "infinite",
                    "number",
                    "notanumber",
                    "normal",
                    "subnormal",
                ],
            ),
            ("type", false, &["f32", "f64"]),
        ],
    ),
    (
        "tirx.ptx.xor",
        &[("type", false, &["pred", "b16", "b32", "b64"])],
    ),
];
