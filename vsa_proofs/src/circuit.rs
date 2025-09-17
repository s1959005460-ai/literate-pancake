// vsa_proofs/src/circuit.rs
use bellman::{Circuit, ConstraintSystem, SynthesisError};
use bellman::gadgets::num::AllocatedNum;
use bls12_381::Scalar;

pub struct VsaCircuit {
    pub x: Option<Scalar>,
    pub y: Option<Scalar>,
}

impl Circuit<Scalar> for VsaCircuit {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
            self.x.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
            self.y.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // 简单约束：y = x^2
        let x_sq = x.square(cs.namespace(|| "x^2"))?;
        cs.enforce(
            || "enforce y = x^2",
            |lc| lc + x_sq.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + y.get_variable(),
        );

        Ok(())
    }
}
