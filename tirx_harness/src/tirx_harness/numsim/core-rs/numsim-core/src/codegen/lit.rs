//! Print any `Serialize` program value as a Rust constant expression with
//! fully qualified enum paths (`AddrSpace::Shared`, `Operand::Reg(Reg(3))`,
//! `Ty { elem: Dtype::F32, lanes: 1u8 }`).
//!
//! Driving the printer through serde keeps it in lockstep with the contract
//! types: a new variant or field needs no printer change. The generated
//! crate imports `program::*`, `dtype::*` and `Domain`, so bare type names
//! resolve. Sequences print as `[..]` (callers add `&` for slice arguments);
//! `Box`/`String` inside a value are not supported (instructions carrying
//! them are passed by reference into `Program::code` instead).

use serde::ser::{self, Serialize};
use std::fmt::{self, Write};

#[derive(Debug)]
pub struct LitError(pub String);

impl fmt::Display for LitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for LitError {}
impl ser::Error for LitError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        LitError(msg.to_string())
    }
}

/// `value` as a Rust expression.
pub fn lit<T: Serialize + ?Sized>(value: &T) -> String {
    let mut p = Printer { out: String::new() };
    value.serialize(&mut p).unwrap_or_else(|e| panic!("codegen literal: {e}"));
    p.out
}

struct Printer {
    out: String,
}

type R = Result<(), LitError>;

/// Comma-separated compound (`[..]`, `(..)`, `Name { .. }`).
struct Compound<'a> {
    p: &'a mut Printer,
    first: bool,
    close: &'static str,
}

impl Compound<'_> {
    fn sep(&mut self) {
        if !self.first {
            self.p.out.push_str(", ");
        }
        self.first = false;
    }
    fn elem<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.sep();
        v.serialize(&mut *self.p)
    }
    fn field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> R {
        self.sep();
        self.p.out.push_str(key);
        self.p.out.push_str(": ");
        v.serialize(&mut *self.p)
    }
    fn end(self) -> R {
        self.p.out.push_str(self.close);
        Ok(())
    }
}

macro_rules! int {
    ($($f:ident $t:ty),*) => {$(
        fn $f(self, v: $t) -> R {
            write!(self.out, concat!("{}", stringify!($t)), v).unwrap();
            Ok(())
        }
    )*};
}

impl<'a> ser::Serializer for &'a mut Printer {
    type Ok = ();
    type Error = LitError;
    type SerializeSeq = Compound<'a>;
    type SerializeTuple = Compound<'a>;
    type SerializeTupleStruct = Compound<'a>;
    type SerializeTupleVariant = Compound<'a>;
    type SerializeMap = ser::Impossible<(), LitError>;
    type SerializeStruct = Compound<'a>;
    type SerializeStructVariant = Compound<'a>;

    fn serialize_bool(self, v: bool) -> R {
        self.out.push_str(if v { "true" } else { "false" });
        Ok(())
    }
    int!(serialize_i8 i8, serialize_i16 i16, serialize_i32 i32, serialize_u8 u8, serialize_u16 u16,
         serialize_u32 u32, serialize_u64 u64, serialize_u128 u128);
    fn serialize_i64(self, v: i64) -> R {
        // `-9223372036854775808i64` is accepted, but keep it unambiguous.
        if v == i64::MIN {
            self.out.push_str("i64::MIN");
        } else {
            write!(self.out, "{v}i64").unwrap();
        }
        Ok(())
    }
    fn serialize_i128(self, v: i128) -> R {
        if v == i128::MIN {
            self.out.push_str("i128::MIN");
        } else {
            write!(self.out, "{v}i128").unwrap();
        }
        Ok(())
    }
    fn serialize_f32(self, v: f32) -> R {
        write!(self.out, "f32::from_bits({:#x})", v.to_bits()).unwrap();
        Ok(())
    }
    fn serialize_f64(self, v: f64) -> R {
        write!(self.out, "f64::from_bits({:#x})", v.to_bits()).unwrap();
        Ok(())
    }
    fn serialize_char(self, v: char) -> R {
        write!(self.out, "{v:?}").unwrap();
        Ok(())
    }
    fn serialize_str(self, v: &str) -> R {
        write!(self.out, "{v:?}").unwrap();
        Ok(())
    }
    fn serialize_bytes(self, v: &[u8]) -> R {
        write!(self.out, "{v:?}").unwrap();
        Ok(())
    }
    fn serialize_none(self) -> R {
        self.out.push_str("None");
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> R {
        self.out.push_str("Some(");
        v.serialize(&mut *self)?;
        self.out.push(')');
        Ok(())
    }
    fn serialize_unit(self) -> R {
        self.out.push_str("()");
        Ok(())
    }
    fn serialize_unit_struct(self, name: &'static str) -> R {
        self.out.push_str(name);
        Ok(())
    }
    fn serialize_unit_variant(self, name: &'static str, _i: u32, variant: &'static str) -> R {
        write!(self.out, "{name}::{variant}").unwrap();
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(self, name: &'static str, v: &T) -> R {
        write!(self.out, "{name}(").unwrap();
        v.serialize(&mut *self)?;
        self.out.push(')');
        Ok(())
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(self, name: &'static str, _i: u32, variant: &'static str, v: &T) -> R {
        write!(self.out, "{name}::{variant}(").unwrap();
        v.serialize(&mut *self)?;
        self.out.push(')');
        Ok(())
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Compound<'a>, LitError> {
        self.out.push('[');
        Ok(Compound { p: self, first: true, close: "]" })
    }
    fn serialize_tuple(self, _len: usize) -> Result<Compound<'a>, LitError> {
        self.out.push('(');
        // A 1-tuple needs a trailing comma; serde only emits tuples >= 2 here.
        Ok(Compound { p: self, first: true, close: ")" })
    }
    fn serialize_tuple_struct(self, name: &'static str, _len: usize) -> Result<Compound<'a>, LitError> {
        write!(self.out, "{name}(").unwrap();
        Ok(Compound { p: self, first: true, close: ")" })
    }
    fn serialize_tuple_variant(self, name: &'static str, _i: u32, variant: &'static str, _len: usize) -> Result<Compound<'a>, LitError> {
        write!(self.out, "{name}::{variant}(").unwrap();
        Ok(Compound { p: self, first: true, close: ")" })
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, LitError> {
        Err(LitError("maps are not printable as Rust literals".into()))
    }
    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<Compound<'a>, LitError> {
        write!(self.out, "{name} {{ ").unwrap();
        Ok(Compound { p: self, first: true, close: " }" })
    }
    fn serialize_struct_variant(self, name: &'static str, _i: u32, variant: &'static str, _len: usize) -> Result<Compound<'a>, LitError> {
        write!(self.out, "{name}::{variant} {{ ").unwrap();
        Ok(Compound { p: self, first: true, close: " }" })
    }
}

impl ser::SerializeSeq for Compound<'_> {
    type Ok = ();
    type Error = LitError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.elem(v)
    }
    fn end(self) -> R {
        Compound::end(self)
    }
}
impl ser::SerializeTuple for Compound<'_> {
    type Ok = ();
    type Error = LitError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.elem(v)
    }
    fn end(self) -> R {
        Compound::end(self)
    }
}
impl ser::SerializeTupleStruct for Compound<'_> {
    type Ok = ();
    type Error = LitError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.elem(v)
    }
    fn end(self) -> R {
        Compound::end(self)
    }
}
impl ser::SerializeTupleVariant for Compound<'_> {
    type Ok = ();
    type Error = LitError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.elem(v)
    }
    fn end(self) -> R {
        Compound::end(self)
    }
}
impl ser::SerializeStruct for Compound<'_> {
    type Ok = ();
    type Error = LitError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> R {
        self.field(key, v)
    }
    fn end(self) -> R {
        Compound::end(self)
    }
}
impl ser::SerializeStructVariant for Compound<'_> {
    type Ok = ();
    type Error = LitError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> R {
        self.field(key, v)
    }
    fn end(self) -> R {
        Compound::end(self)
    }
}

#[cfg(test)]
mod tests {
    use super::lit;
    use crate::dtype::Ty;
    use crate::program::*;

    #[test]
    fn qualified_paths() {
        assert_eq!(lit(&Operand::Reg(Reg(3))), "Operand::Reg(Reg(3u32))");
        assert_eq!(lit(&AddrSpace::Shared), "AddrSpace::Shared");
        assert_eq!(lit(&Ty::F32), "Ty { elem: Dtype::F32, lanes: 1u8 }");
        assert_eq!(lit(&Some(Reg(1))), "Some(Reg(1u32))");
        assert_eq!(lit(&[Reg(1), Reg(2)][..]), "[Reg(1u32), Reg(2u32)]");
        assert_eq!(
            lit(&PhaseArg::Parity(Operand::Const(ConstId(0)))),
            "PhaseArg::Parity(Operand::Const(ConstId(0u32)))"
        );
        assert_eq!(lit(&FenceKind::TensormapAcquire { addr: Operand::Reg(Reg(0)), space: AddrSpace::Global }),
            "FenceKind::TensormapAcquire { addr: Operand::Reg(Reg(0u32)), space: AddrSpace::Global }");
        assert_eq!(lit(&Some((AtomOp::Add, crate::Dtype::F32))), "Some((AtomOp::Add, Dtype::F32))");
    }
}
