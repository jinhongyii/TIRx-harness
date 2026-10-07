//! Spelling-level PTX `cvt` entry point: parse one exact `cvt.*` spelling and
//! dispatch it to the plain functions in `cvt::{int, float, narrow}`.
//!
//! Replaces the legacy `variant::Cvt<Src, Dst, Mode>` / `CvtMode` /
//! `PackedMode` marker table in `engine-rs/src/runtime/instructions/reg.rs`
//! (cvt section, ~lines 3065-5140, plus `cvt_pack_variant!` at ~1543).  The
//! legality rules below mirror which marker combinations legacy registered:
//! a spelling with no legacy specialization is rejected.

use crate::cvt::float::*;
use crate::cvt::int::*;
use crate::cvt::narrow::*;
use crate::cvt::pack::ptx_cvt_pack;
use crate::cvt::ptx::{PtxFloatRounding, PtxIntegerRounding};
use crate::scalar::LowPrecisionFormat;
use crate::types::{OpError, OpResult};

/// PTX `cvt` rounding modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CvtRounding {
    Rn,
    Rz,
    Rm,
    Rp,
    Rna,
    Rs,
    Rni,
    Rzi,
    Rmi,
    Rpi,
}

impl CvtRounding {
    fn parse(token: &str) -> Option<Self> {
        Some(match token {
            "rn" => Self::Rn,
            "rz" => Self::Rz,
            "rm" => Self::Rm,
            "rp" => Self::Rp,
            "rna" => Self::Rna,
            "rs" => Self::Rs,
            "rni" => Self::Rni,
            "rzi" => Self::Rzi,
            "rmi" => Self::Rmi,
            "rpi" => Self::Rpi,
            _ => return None,
        })
    }

    /// The `.irnd` mode, if this is one.
    pub fn integer(self) -> Option<PtxIntegerRounding> {
        Some(match self {
            Self::Rni => PtxIntegerRounding::NearestEven,
            Self::Rzi => PtxIntegerRounding::Zero,
            Self::Rmi => PtxIntegerRounding::NegativeInfinity,
            Self::Rpi => PtxIntegerRounding::PositiveInfinity,
            _ => return None,
        })
    }

    /// The `.frnd` mode (`.rn/.rz/.rm/.rp/.rna`), if this is one.
    pub fn float(self) -> Option<PtxFloatRounding> {
        Some(match self {
            Self::Rn => PtxFloatRounding::NearestEven,
            Self::Rz => PtxFloatRounding::Zero,
            Self::Rm => PtxFloatRounding::NegativeInfinity,
            Self::Rp => PtxFloatRounding::PositiveInfinity,
            Self::Rna => PtxFloatRounding::NearestAway,
            _ => return None,
        })
    }
}

/// `.scaled::*::ue8m0` modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CvtScale {
    /// `.scaled::n1::ue8m0`: one `u8` scale for both packed outputs.
    N1,
    /// `.scaled::n2::ue8m0`: a `u16` holding two scale bytes.
    N2,
}

/// A `cvt` source or destination type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CvtType {
    Int(IntKind),
    F16,
    Bf16,
    F32,
    F64,
    Tf32,
    F16x2,
    Bf16x2,
    NarrowX2(NarrowKind),
    NarrowX4(NarrowKind),
    S2f6x2,
    Ue8m0x2,
}

impl CvtType {
    /// Parse a PTX type suffix.
    pub fn parse(name: &str) -> Option<Self> {
        if let Some(kind) = IntKind::from_ptx(name) {
            return Some(Self::Int(kind));
        }
        let narrow = |stem: &str| -> Option<NarrowKind> {
            Some(match stem {
                "e4m3" => NarrowKind::E4m3,
                "e5m2" => NarrowKind::E5m2,
                "e2m1" => NarrowKind::E2m1,
                "e2m3" => NarrowKind::E2m3,
                "e3m2" => NarrowKind::E3m2,
                "ue5m3" => NarrowKind::Ue5m3,
                _ => return None,
            })
        };
        Some(match name {
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "tf32" => Self::Tf32,
            "f16x2" => Self::F16x2,
            "bf16x2" => Self::Bf16x2,
            "s2f6x2" => Self::S2f6x2,
            "ue8m0x2" => Self::Ue8m0x2,
            _ => {
                if let Some(stem) = name.strip_suffix("x2") {
                    Self::NarrowX2(narrow(stem)?)
                } else if let Some(stem) = name.strip_suffix("x4") {
                    // `.ue5m3` has no four-element form.
                    Self::NarrowX4(narrow(stem).filter(|kind| *kind != NarrowKind::Ue5m3)?)
                } else {
                    return None;
                }
            }
        })
    }

    /// Bit width of the register payload.
    pub fn bits(self) -> u32 {
        match self {
            Self::Int(kind) => kind.bits(),
            Self::F16 | Self::Bf16 => 16,
            Self::F32 | Self::Tf32 | Self::F16x2 | Self::Bf16x2 => 32,
            Self::F64 => 64,
            Self::NarrowX2(NarrowKind::E2m1) => 8,
            Self::NarrowX2(_) | Self::S2f6x2 | Self::Ue8m0x2 => 16,
            Self::NarrowX4(NarrowKind::E2m1) => 16,
            Self::NarrowX4(_) => 32,
        }
    }

    fn mask(self) -> u64 {
        if self.bits() == 64 {
            u64::MAX
        } else {
            (1_u64 << self.bits()) - 1
        }
    }
}

/// Operands of one `cvt` execution, as raw register payloads.
///
/// * scalar forms read `a`;
/// * two-primary forms (`d, a, b`) read `a` (upper element) and `b`;
/// * `.rs` four-packs read `a, b, c, d` (`a` in the most significant field)
///   and `rbits`; `.rs.{f16x2,bf16x2}` read `a, b, rbits`;
/// * `.scaled` forms read `scale` (`u8` for n1, `u16` for n2);
/// * `cvt.pack` reads `a`, `b` (as `s32`) and `c` (`b32`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CvtOperands {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
    pub scale: u16,
    pub rbits: u32,
}

impl CvtOperands {
    /// A single-source operand set.
    pub const fn unary(a: u64) -> Self {
        Self {
            a,
            b: 0,
            c: 0,
            d: 0,
            scale: 0,
            rbits: 0,
        }
    }

    /// A two-primary operand set (`a` is the upper element).
    pub const fn pair(a: u64, b: u64) -> Self {
        Self {
            a,
            b,
            c: 0,
            d: 0,
            scale: 0,
            rbits: 0,
        }
    }
}

/// One parsed `cvt` spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CvtSpelling {
    pub rounding: Option<CvtRounding>,
    pub ftz: bool,
    pub sat: bool,
    pub satfinite: bool,
    pub relu: bool,
    pub pzo: bool,
    pub scale: Option<CvtScale>,
    pub dst: CvtType,
    pub src: CvtType,
    /// `cvt.pack.sat` field `(bits, signed)`, when this is a pack form.
    pub pack: Option<(u32, bool)>,
}

fn reject<T>(spelling: &CvtSpelling, why: &str) -> OpResult<T> {
    Err(OpError::message(format!(
        "unsupported cvt form {spelling:?}: {why}"
    )))
}

impl CvtSpelling {
    /// Parse a spelling such as `cvt.rn.satfinite.relu.e4m3x2.f32`.
    /// Modifiers may appear in any order; each at most once.
    pub fn parse(spelling: &str) -> OpResult<Self> {
        let error = |why: &str| OpError::message(format!("cannot parse `{spelling}`: {why}"));
        let tokens: Vec<&str> = spelling.split('.').collect();
        if tokens.first() != Some(&"cvt") || tokens.len() < 3 {
            return Err(error("expected cvt.<modifiers>.<dtype>.<atype>"));
        }
        if tokens[1] == "pack" {
            return Self::parse_pack(&tokens).ok_or_else(|| error("bad cvt.pack form"));
        }
        let n = tokens.len();
        let dst = CvtType::parse(tokens[n - 2]).ok_or_else(|| error("unknown destination type"))?;
        let src = CvtType::parse(tokens[n - 1]).ok_or_else(|| error("unknown source type"))?;
        let mut parsed = Self {
            rounding: None,
            ftz: false,
            sat: false,
            satfinite: false,
            relu: false,
            pzo: false,
            scale: None,
            dst,
            src,
            pack: None,
        };
        for &token in &tokens[1..n - 2] {
            let duplicate = match token {
                "ftz" => std::mem::replace(&mut parsed.ftz, true),
                "sat" => std::mem::replace(&mut parsed.sat, true),
                "satfinite" => std::mem::replace(&mut parsed.satfinite, true),
                "relu" => std::mem::replace(&mut parsed.relu, true),
                "pzo" => std::mem::replace(&mut parsed.pzo, true),
                "scaled::n1::ue8m0" => parsed.scale.replace(CvtScale::N1).is_some(),
                "scaled::n2::ue8m0" => parsed.scale.replace(CvtScale::N2).is_some(),
                other => {
                    let rounding =
                        CvtRounding::parse(other).ok_or_else(|| error("unknown modifier"))?;
                    parsed.rounding.replace(rounding).is_some()
                }
            };
            if duplicate {
                return Err(error("repeated modifier"));
            }
        }
        Ok(parsed)
    }

    fn parse_pack(tokens: &[&str]) -> Option<Self> {
        // cvt.pack.sat.{u16,s16}.s32  |  cvt.pack.sat.{u8,s8,u4,s4,u2,s2}.s32.b32
        if tokens.get(2) != Some(&"sat") || tokens.get(4) != Some(&"s32") {
            return None;
        }
        let field = tokens.get(3)?;
        match (*field, tokens.len()) {
            ("u16" | "s16", 5) => {}
            ("u8" | "s8" | "u4" | "s4" | "u2" | "s2", 6) if tokens[5] == "b32" => {}
            _ => return None,
        }
        let signed = field.starts_with('s');
        let bits: u32 = field[1..].parse().ok()?;
        Some(Self {
            rounding: None,
            ftz: false,
            sat: true,
            satfinite: false,
            relu: false,
            pzo: false,
            scale: None,
            dst: CvtType::Int(IntKind::U32),
            src: CvtType::Int(IntKind::S32),
            pack: Some((bits, signed)),
        })
    }

    /// Execute this form on `operands`; returns the destination payload.
    pub fn execute(&self, operands: &CvtOperands) -> OpResult<u64> {
        if let Some((bits, signed)) = self.pack {
            return Ok(u64::from(execute_pack(bits, signed, operands)));
        }
        let a = operands.a & self.src.mask();
        let result = self.dispatch(a, operands)?;
        Ok(result & self.dst.mask())
    }

    fn no_packed_modifiers(&self) -> bool {
        !self.satfinite && !self.relu && !self.pzo && self.scale.is_none()
    }

    fn dispatch(&self, a: u64, ops: &CvtOperands) -> OpResult<u64> {
        use CvtType as T;
        match (self.src, self.dst) {
            (T::Int(src), T::Int(dst)) => {
                if self.rounding.is_some() || self.ftz || !self.no_packed_modifiers() {
                    return reject(self, "integer conversions take only .sat");
                }
                Ok(cvt_int_to_int(a, src, dst, self.sat))
            }
            (T::F16 | T::Bf16 | T::F32 | T::F64, T::Int(dst)) => self.float_to_int(a, dst),
            (T::Int(src), T::F16 | T::Bf16 | T::F32 | T::F64) => self.int_to_float(a, src),
            (T::F16 | T::Bf16 | T::F32 | T::F64, T::F16 | T::Bf16 | T::F32 | T::F64 | T::Tf32) => {
                self.float_to_float(a)
            }
            (T::F32, T::F16x2 | T::Bf16x2) => self.f32_pair_to_half2(ops),
            (T::F32 | T::F16x2 | T::Bf16x2, T::NarrowX2(kind)) => {
                self.pack_narrow_x2(self.pair_source(ops)?, kind, ops.scale)
            }
            (T::F32, T::NarrowX4(kind)) => self.pack_narrow_x4(ops, kind),
            (T::NarrowX2(kind), T::F16x2 | T::Bf16x2) => {
                self.unpack_narrow_x2(a as u16, kind, ops.scale)
            }
            (T::F32 | T::Bf16x2, T::S2f6x2) => {
                if self.rounding != Some(CvtRounding::Rn)
                    || !self.satfinite
                    || self.pzo
                    || self.ftz
                    || self.sat
                    || self.scale == Some(CvtScale::N1)
                {
                    return reject(
                        self,
                        "s2f6x2 packs are cvt.rn.satfinite{.relu}{.scaled::n2}",
                    );
                }
                let scale = self.scale.map(|_| ops.scale);
                Ok(u64::from(cvt_pack_s2f6x2(
                    self.pair_source(ops)?,
                    scale,
                    self.relu,
                )))
            }
            (T::S2f6x2, T::Bf16x2) => {
                if self.rounding != Some(CvtRounding::Rn)
                    || self.pzo
                    || self.ftz
                    || self.sat
                    || self.scale == Some(CvtScale::N1)
                {
                    return reject(
                        self,
                        "s2f6x2 unpack is cvt.rn{.relu}{.satfinite}{.scaled::n2}",
                    );
                }
                let scale = self.scale.map(|_| ops.scale);
                Ok(u64::from(cvt_unpack_s2f6x2(
                    a as u16,
                    scale,
                    self.relu,
                    self.satfinite,
                )))
            }
            (T::F32 | T::Bf16x2, T::Ue8m0x2) => {
                let round_up = match self.rounding {
                    Some(CvtRounding::Rz) => false,
                    Some(CvtRounding::Rp) => true,
                    _ => return reject(self, "ue8m0x2 takes .rz or .rp"),
                };
                if self.relu || self.pzo || self.ftz || self.sat || self.scale.is_some() {
                    return reject(self, "ue8m0x2 takes only .satfinite");
                }
                Ok(u64::from(cvt_pack_ue8m0x2(
                    self.pair_source(ops)?,
                    round_up,
                    self.satfinite,
                )?))
            }
            (T::Ue8m0x2, T::Bf16x2) => {
                if self.rounding != Some(CvtRounding::Rn)
                    || self.ftz
                    || self.sat
                    || !self.no_packed_modifiers()
                {
                    return reject(self, "the only spelling is cvt.rn.bf16x2.ue8m0x2");
                }
                Ok(u64::from(cvt_unpack_ue8m0x2(a as u16)))
            }
            _ => reject(self, "no such source/destination pair"),
        }
    }

    fn pair_source(&self, ops: &CvtOperands) -> OpResult<PairSource> {
        Ok(match self.src {
            CvtType::F32 => PairSource::F32 {
                high: f32::from_bits(ops.a as u32),
                low: f32::from_bits(ops.b as u32),
            },
            CvtType::F16x2 => PairSource::F16x2(ops.a as u32),
            CvtType::Bf16x2 => PairSource::Bf16x2(ops.a as u32),
            _ => return reject(self, "not a pair source"),
        })
    }

    fn float_to_int(&self, a: u64, dst: IntKind) -> OpResult<u64> {
        let Some(mode) = self.rounding.and_then(CvtRounding::integer) else {
            return reject(self, "float-to-integer requires .rni/.rzi/.rmi/.rpi");
        };
        if !self.no_packed_modifiers() || (self.ftz && self.src != CvtType::F32) {
            return reject(self, "illegal modifier");
        }
        Ok(match self.src {
            CvtType::F32 => cvt_f32_to_int(f32::from_bits(a as u32), mode, self.ftz, self.sat, dst),
            CvtType::F64 => cvt_f64_to_int(f64::from_bits(a), mode, self.sat, dst),
            CvtType::F16 => cvt_f16_to_int(a as u16, mode, self.sat, dst),
            CvtType::Bf16 => {
                if dst.bits() == 8 {
                    return reject(self, "bf16 has no 8-bit integer destination");
                }
                cvt_bf16_to_int(a as u16, mode, self.sat, dst)
            }
            _ => unreachable!(),
        })
    }

    fn int_to_float(&self, a: u64, src: IntKind) -> OpResult<u64> {
        let rounding = match self.rounding {
            None => PtxFloatRounding::NearestEven,
            Some(r @ (CvtRounding::Rn | CvtRounding::Rz | CvtRounding::Rm | CvtRounding::Rp)) => {
                r.float().unwrap()
            }
            Some(_) => return reject(self, "integer-to-float takes .rn/.rz/.rm/.rp"),
        };
        if !self.no_packed_modifiers() || (self.ftz && self.dst != CvtType::F32) {
            return reject(self, "illegal modifier");
        }
        if self.dst == CvtType::Bf16 && (src.bits() == 8 || self.sat) {
            return reject(self, "no such bf16 integer form");
        }
        if self.sat {
            let one = cvt_int_sat_is_one(a, src);
            return Ok(match self.dst {
                CvtType::F16 => u64::from(u16::from(one) * 0x3c00),
                CvtType::F32 => u64::from(f32::from(u8::from(one)).to_bits()),
                CvtType::F64 => f64::from(u8::from(one)).to_bits(),
                _ => unreachable!(),
            });
        }
        Ok(match self.dst {
            CvtType::F32 => u64::from(cvt_int_to_f32(a, src, rounding).to_bits()),
            CvtType::F64 => cvt_int_to_f64(a, src, rounding).to_bits(),
            CvtType::F16 => u64::from(cvt_int_to_f16(a, src, rounding)),
            CvtType::Bf16 => u64::from(cvt_int_to_bf16(a, src, rounding)),
            _ => unreachable!(),
        })
    }

    fn float_to_float(&self, a: u64) -> OpResult<u64> {
        use CvtType as T;
        let half = |t: CvtType| match t {
            T::F16 => Some(LowPrecisionFormat::F16),
            T::Bf16 => Some(LowPrecisionFormat::Bf16),
            _ => None,
        };
        let irnd = self.rounding.and_then(CvtRounding::integer);
        let frnd = self.rounding.and_then(CvtRounding::float);
        let only_irnd = self.rounding.is_none() || irnd.is_some();
        let directed = matches!(
            self.rounding,
            Some(CvtRounding::Rn | CvtRounding::Rz | CvtRounding::Rm | CvtRounding::Rp)
        );
        let packed_free = self.no_packed_modifiers();
        match (self.src, self.dst) {
            (T::F32, T::F32) if only_irnd && packed_free => Ok(u64::from(
                cvt_f32_to_f32(f32::from_bits(a as u32), irnd, self.ftz, self.sat).to_bits(),
            )),
            (T::F64, T::F64) if only_irnd && packed_free && !self.ftz => {
                Ok(cvt_f64_to_f64(f64::from_bits(a), irnd, self.sat).to_bits())
            }
            (T::F32, T::F64) if self.rounding.is_none() && packed_free => {
                Ok(cvt_f32_to_f64(f32::from_bits(a as u32), self.ftz, self.sat).to_bits())
            }
            (T::F64, T::F32) if directed && packed_free => Ok(u64::from(
                cvt_f64_to_f32(f64::from_bits(a), frnd.unwrap(), self.ftz, self.sat).to_bits(),
            )),
            (T::F16, T::F16) | (T::Bf16, T::Bf16)
                if only_irnd && packed_free && !self.ftz && !(self.sat && self.src == T::Bf16) =>
            {
                Ok(u64::from(cvt_half_to_half(
                    a as u16,
                    half(self.src).unwrap(),
                    irnd,
                    self.sat,
                )))
            }
            (T::F16 | T::Bf16, T::F32)
                if self.rounding.is_none() && packed_free && !(self.sat && self.src == T::Bf16) =>
            {
                Ok(u64::from(
                    cvt_half_to_f32(a as u16, half(self.src).unwrap(), self.ftz, self.sat)
                        .to_bits(),
                ))
            }
            (T::F16 | T::Bf16, T::F64)
                if self.rounding.is_none()
                    && packed_free
                    && !self.ftz
                    && !(self.sat && self.src == T::Bf16) =>
            {
                Ok(cvt_half_to_f64(a as u16, half(self.src).unwrap(), self.sat).to_bits())
            }
            (T::F32, T::F16 | T::Bf16) => self.f32_to_half(a as u32, half(self.dst).unwrap()),
            (T::F32, T::Tf32) => {
                let ok = match self.rounding {
                    Some(CvtRounding::Rn | CvtRounding::Rz) => true,
                    Some(CvtRounding::Rna) => !self.relu && !self.pzo,
                    _ => false,
                };
                if !ok || self.ftz || self.sat || self.scale.is_some() {
                    return reject(
                        self,
                        "tf32 takes .rn/.rz{.relu}{.satfinite}{.pzo} or .rna{.satfinite}",
                    );
                }
                Ok(u64::from(cvt_f32_to_tf32(
                    f32::from_bits(a as u32),
                    frnd.unwrap(),
                    self.relu,
                    self.satfinite,
                    self.pzo,
                )))
            }
            (T::F64, T::F16 | T::Bf16)
                if directed && packed_free && !self.ftz && !(self.sat && self.dst == T::Bf16) =>
            {
                Ok(u64::from(cvt_f64_to_half(
                    f64::from_bits(a),
                    half(self.dst).unwrap(),
                    frnd.unwrap(),
                    self.sat,
                )))
            }
            (T::F16, T::Bf16) | (T::Bf16, T::F16)
                if directed && packed_free && !self.ftz && !self.sat =>
            {
                Ok(u64::from(cvt_half_cross(
                    a as u16,
                    half(self.src).unwrap(),
                    frnd.unwrap(),
                )))
            }
            _ => reject(self, "no such float conversion"),
        }
    }

    fn f32_to_half(&self, a: u32, format: LowPrecisionFormat) -> OpResult<u64> {
        let near = matches!(self.rounding, Some(CvtRounding::Rn | CvtRounding::Rz));
        let directed = near || matches!(self.rounding, Some(CvtRounding::Rm | CvtRounding::Rp));
        let packed = self.relu || self.satfinite || self.pzo;
        let legal = directed
            && self.scale.is_none()
            && !(packed && (!near || self.ftz || self.sat))
            && !(self.sat && matches!(format, LowPrecisionFormat::Bf16));
        if !legal {
            return reject(self, "illegal f32 narrowing modifiers");
        }
        let modifiers = HalfNarrowing {
            ftz: self.ftz,
            sat: self.sat,
            relu: self.relu,
            satfinite: self.satfinite,
            pzo: self.pzo,
        };
        let rounding = self.rounding.and_then(CvtRounding::float).unwrap();
        Ok(u64::from(cvt_f32_to_half(
            f32::from_bits(a),
            format,
            rounding,
            modifiers,
        )))
    }

    fn f32_pair_to_half2(&self, ops: &CvtOperands) -> OpResult<u64> {
        let format = match self.dst {
            CvtType::F16x2 => LowPrecisionFormat::F16,
            _ => LowPrecisionFormat::Bf16,
        };
        if self.ftz || self.sat || self.scale.is_some() {
            return reject(self, "packed half takes .relu/.satfinite/.pzo only");
        }
        let high = f32::from_bits(ops.a as u32);
        let low = f32::from_bits(ops.b as u32);
        match self.rounding {
            Some(r @ (CvtRounding::Rn | CvtRounding::Rz)) => Ok(u64::from(cvt_f32_pair_to_half2(
                high,
                low,
                format,
                r.float().unwrap(),
                self.relu,
                self.satfinite,
                self.pzo,
            ))),
            Some(CvtRounding::Rs) if !self.pzo => Ok(u64::from(cvt_f32_pair_to_half2_rs(
                high,
                low,
                ops.rbits,
                format,
                self.relu,
                self.satfinite,
            )?)),
            _ => reject(self, "packed half takes .rn/.rz/.rs"),
        }
    }

    fn pack_narrow_x2(&self, source: PairSource, kind: NarrowKind, scale: u16) -> OpResult<u64> {
        if self.ftz || self.sat || self.scale == Some(CvtScale::N2) {
            return reject(self, "illegal modifier");
        }
        let scale = self.scale.map(|_| scale as u8);
        let rounding = self.rounding;
        if kind == NarrowKind::Ue5m3 {
            if self.relu || self.pzo {
                return reject(self, "ue5m3x2 has no .relu/.pzo");
            }
            let ok = match rounding {
                Some(CvtRounding::Rn | CvtRounding::Rz) => true,
                Some(CvtRounding::Rp) => scale.is_none(),
                _ => false,
            };
            if !ok {
                return reject(self, "ue5m3x2 takes .rn/.rz (scaled) or .rn/.rz/.rp");
            }
            let mode = rounding.unwrap().float().unwrap();
            return Ok(u64::from(if self.satfinite {
                cvt_pack_narrow_x2_rounded(source, kind, mode, false, false, scale)
            } else {
                cvt_pack_ue5m3x2_unsaturated(source, mode, scale)
            }));
        }
        if !self.satfinite {
            return reject(self, "signed narrow packs require .satfinite");
        }
        let mode = match rounding {
            Some(CvtRounding::Rn) => PtxFloatRounding::NearestEven,
            Some(CvtRounding::Rz) => PtxFloatRounding::Zero,
            _ => return reject(self, "signed narrow packs take .rn or .rz"),
        };
        let legacy_rn = matches!(kind, NarrowKind::E4m3 | NarrowKind::E5m2 | NarrowKind::E2m1)
            && mode == PtxFloatRounding::NearestEven
            && scale.is_none()
            && !self.pzo;
        Ok(u64::from(if legacy_rn {
            cvt_pack_narrow_x2_rn(source, kind, self.relu)
        } else {
            cvt_pack_narrow_x2_rounded(source, kind, mode, self.relu, self.pzo, scale)
        }))
    }

    fn pack_narrow_x4(&self, ops: &CvtOperands, kind: NarrowKind) -> OpResult<u64> {
        if self.rounding != Some(CvtRounding::Rs)
            || !self.satfinite
            || self.pzo
            || self.ftz
            || self.sat
            || self.scale.is_some()
        {
            return reject(self, "x4 packs are cvt.rs{.relu}.satfinite");
        }
        let values = [ops.a, ops.b, ops.c, ops.d].map(|bits| f32::from_bits(bits as u32));
        Ok(u64::from(cvt_pack_narrow_x4_rs(
            values, ops.rbits, kind, self.relu,
        )))
    }

    fn unpack_narrow_x2(&self, bits: u16, kind: NarrowKind, scale: u16) -> OpResult<u64> {
        if self.rounding != Some(CvtRounding::Rn)
            || self.pzo
            || self.ftz
            || self.sat
            || self.scale == Some(CvtScale::N1)
            || (kind == NarrowKind::Ue5m3 && self.relu)
        {
            return reject(
                self,
                "narrow unpack is cvt.rn{.relu}{.satfinite}{.scaled::n2}",
            );
        }
        if self.dst == CvtType::F16x2 {
            if self.satfinite || self.scale.is_some() {
                return reject(self, "f16x2 unpack has no .satfinite/.scaled");
            }
            return Ok(u64::from(cvt_unpack_narrow_x2_f16x2(bits, kind, self.relu)));
        }
        Ok(u64::from(cvt_unpack_narrow_x2_bf16x2(
            bits,
            kind,
            self.relu,
            self.satfinite,
            self.scale.map(|_| scale),
        )))
    }
}

fn execute_pack(bits: u32, signed: bool, ops: &CvtOperands) -> u32 {
    let (a, b, c) = (ops.a as u32 as i32, ops.b as u32 as i32, ops.c as u32);
    match (bits, signed) {
        (16, false) => ptx_cvt_pack::<16, false>(a, b, 0),
        (16, true) => ptx_cvt_pack::<16, true>(a, b, 0),
        (8, false) => ptx_cvt_pack::<8, false>(a, b, c),
        (8, true) => ptx_cvt_pack::<8, true>(a, b, c),
        (4, false) => ptx_cvt_pack::<4, false>(a, b, c),
        (4, true) => ptx_cvt_pack::<4, true>(a, b, c),
        (2, false) => ptx_cvt_pack::<2, false>(a, b, c),
        _ => ptx_cvt_pack::<2, true>(a, b, c),
    }
}

/// Parse `spelling` and execute it on `operands`; returns the destination
/// payload (low `BitWidth(dst)` bits).
pub fn ptx_cvt(spelling: &str, operands: &CvtOperands) -> OpResult<u64> {
    CvtSpelling::parse(spelling)?.execute(operands)
}

/// Single-source convenience form of [`ptx_cvt`].
pub fn ptx_cvt_unary(spelling: &str, source: u64) -> OpResult<u64> {
    ptx_cvt(spelling, &CvtOperands::unary(source))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(value: f32) -> u64 {
        u64::from(value.to_bits())
    }

    fn scaled(a: u64, b: u64, scale: u16) -> CvtOperands {
        CvtOperands {
            scale,
            ..CvtOperands::pair(a, b)
        }
    }

    /// Expectations of `tests/numsim/runtime/test_ptx_cvt_94.py`
    /// (`test_ptx94_non_pzo_cvt_families_match_independent_bit_oracle`).
    #[test]
    fn ptx94_families_match_independent_bit_oracle() {
        let cases: [(&str, CvtOperands, u64); 8] = [
            (
                "cvt.rz.satfinite.scaled::n1::ue8m0.e2m3x2.f32",
                scaled(f(2.125), f(-2.375), 128),
                0x0829,
            ),
            (
                "cvt.rz.satfinite.e3m2x2.bf16x2",
                CvtOperands::unary(0x3F88_BF98),
                0x0C2C,
            ),
            (
                "cvt.rp.satfinite.ue5m3x2.f32",
                CvtOperands::pair(f(1.0625), f(1.1875)),
                0x797A,
            ),
            (
                "cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32",
                scaled(f(2.125), f(2.375), 128),
                0x787A,
            ),
            (
                "cvt.rz.satfinite.ue5m3x2.f16x2",
                CvtOperands::unary(0x3C40_3CC0),
                0x7879,
            ),
            (
                "cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.bf16x2",
                scaled(0x4008_4018, 0, 128),
                0x787A,
            ),
            (
                "cvt.rn.f16x2.ue5m3x2",
                CvtOperands::unary(0x0178),
                0x0080_3C00,
            ),
            (
                "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2",
                scaled(0xFE01, 0, 0x707F),
                0x4060_3700,
            ),
        ];
        for (spelling, operands, expected) in cases {
            assert_eq!(
                ptx_cvt(spelling, &operands).unwrap(),
                expected,
                "{spelling}"
            );
        }
    }

    /// `test_ptx94_pzo_normalizes_each_negative_zero_after_narrow_conversion`.
    #[test]
    fn ptx94_pzo_normalizes_each_negative_zero_after_narrow_conversion() {
        let pair = CvtOperands::pair(f(-1e-30), f(-0.0));
        assert_eq!(
            ptx_cvt("cvt.rz.satfinite.e4m3x2.f32", &pair).unwrap(),
            0x8080
        );
        assert_eq!(
            ptx_cvt("cvt.rz.satfinite.pzo.e4m3x2.f32", &pair).unwrap(),
            0
        );
        let packed = scaled(0x8000_8000, 0, 127);
        assert_eq!(
            ptx_cvt("cvt.rn.satfinite.scaled::n1::ue8m0.e2m3x2.bf16x2", &packed).unwrap(),
            0x2020
        );
        assert_eq!(
            ptx_cvt(
                "cvt.rn.satfinite.pzo.scaled::n1::ue8m0.e2m3x2.bf16x2",
                &packed
            )
            .unwrap(),
            0
        );
    }

    /// `tests/numsim/runtime/test_ptx_cvt_pzo.py`.
    #[test]
    fn pzo_is_post_conversion_and_applies_to_each_packed_result() {
        let tiny = f32::from_bits(0x8000_0001);
        let source = [
            -0.0,
            tiny,
            -1.0,
            0.0,
            1.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ];
        let f16 = [0, 0, 0xBC00, 0, 0x3C00, 0xFC00, 0x7C00, 0x7FFF];
        let bf16 = [0, 0, 0xBF80, 0, 0x3F80, 0xFF80, 0x7F80, 0x7FFF];
        let tf32 = [
            0,
            0,
            0xBF80_0000,
            0,
            0x3F80_0000,
            0xFF80_0000,
            0x7F80_0000,
            0x7FFF_E000,
        ];
        for index in 0..source.len() {
            let a = f(source[index]);
            let previous = (index + source.len() - 1) % source.len();
            let pair = CvtOperands::pair(a, f(source[previous]));
            assert_eq!(ptx_cvt_unary("cvt.rz.pzo.f16.f32", a).unwrap(), f16[index]);
            assert_eq!(
                ptx_cvt_unary("cvt.rz.pzo.bf16.f32", a).unwrap(),
                bf16[index]
            );
            assert_eq!(
                ptx_cvt_unary("cvt.rz.pzo.tf32.f32", a).unwrap(),
                tf32[index]
            );
            assert_eq!(
                ptx_cvt("cvt.rz.pzo.f16x2.f32", &pair).unwrap(),
                (f16[index] << 16) | f16[previous]
            );
            assert_eq!(
                ptx_cvt("cvt.rz.pzo.bf16x2.f32", &pair).unwrap(),
                (bf16[index] << 16) | bf16[previous]
            );
        }
    }

    #[test]
    fn cvt_pack_saturates_each_field() {
        let ops = CvtOperands {
            a: 70_000,
            b: (-5_i32) as u32 as u64,
            ..CvtOperands::default()
        };
        assert_eq!(ptx_cvt("cvt.pack.sat.u16.s32", &ops).unwrap(), 0xffff_0000);
        assert_eq!(ptx_cvt("cvt.pack.sat.s16.s32", &ops).unwrap(), 0x7fff_fffb);
        assert!(ptx_cvt("cvt.pack.sat.u8.s32", &ops).is_err());
    }

    #[test]
    fn integer_and_saturating_integer_sources() {
        assert_eq!(ptx_cvt_unary("cvt.u32.s8", 0x80).unwrap(), 0xffff_ff80);
        assert_eq!(ptx_cvt_unary("cvt.sat.u32.s8", 0x80).unwrap(), 0);
        assert_eq!(ptx_cvt_unary("cvt.rn.sat.f16.s32", 7).unwrap(), 0x3c00);
        assert_eq!(ptx_cvt_unary("cvt.rz.sat.f32.s32", 0xffff_ffff).unwrap(), 0);
    }

    #[test]
    fn spellings_without_a_legacy_specialization_are_rejected() {
        for spelling in [
            "cvt.rn.s32.f32",
            "cvt.rzi.s8.bf16",
            "cvt.rzi.ftz.s32.f64",
            "cvt.rn.f32.f32",
            "cvt.rm.relu.f16.f32",
            "cvt.rn.sat.bf16.f32",
            "cvt.rna.relu.tf32.f32",
            "cvt.rn.e4m3x2.f32",
            "cvt.rn.relu.f16x2.ue5m3x2",
            "cvt.rs.e4m3x4.f32",
            "cvt.rn.rn.f16.f32",
            "cvt.rn.f17.f32",
        ] {
            assert!(
                ptx_cvt_unary(spelling, 0).is_err(),
                "{spelling} should be rejected"
            );
        }
    }
}
