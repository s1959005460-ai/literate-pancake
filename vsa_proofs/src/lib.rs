// vsa_proofs/src/lib.rs
mod circuit;

use crate::circuit::VsaCircuit;
use bls12_381::Scalar;
use bellman::groth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof};
use rand::thread_rng;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn prove_and_verify(x: u64) -> bool {
    let rng = &mut thread_rng();

    // 构造电路: y = x^2
    let x_scalar = Scalar::from(x);
    let y_scalar = x_scalar * x_scalar;

    let circuit = VsaCircuit {
        x: Some(x_scalar),
        y: Some(y_scalar),
    };

    let params = generate_random_parameters::<_, _, _>(circuit, rng).unwrap();
    let pvk = prepare_verifying_key(&params.vk);

    // 生成 proof
    let circuit2 = VsaCircuit {
        x: Some(x_scalar),
        y: Some(y_scalar),
    };
    let proof = create_random_proof(circuit2, &params, rng).unwrap();

    // 验证 proof
    verify_proof(&pvk, &proof, &[y_scalar]).unwrap_or(false)
}
