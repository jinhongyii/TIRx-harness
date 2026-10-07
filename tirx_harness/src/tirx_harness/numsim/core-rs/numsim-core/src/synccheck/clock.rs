//! Dense per-warp vector clock (one component per warp of the whole program).

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Clock(Box<[u32]>);

impl Clock {
    pub fn zero(warps: usize) -> Self {
        Self(vec![0; warps].into_boxed_slice())
    }

    pub fn tick(&mut self, warp: usize) {
        self.0[warp] += 1;
    }

    pub fn join(&mut self, other: &Self) {
        for (mine, theirs) in self.0.iter_mut().zip(other.0.iter()) {
            *mine = (*mine).max(*theirs);
        }
    }

    /// `self <= other` componentwise (happens-before or equal).
    pub fn leq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a <= b)
    }

    /// Strict happens-before.
    pub fn hb(&self, other: &Self) -> bool {
        self.leq(other) && self != other
    }
}
