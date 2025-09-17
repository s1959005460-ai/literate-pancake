use hmac::{Hmac, Mac};
use sha2::Sha256;

/// HMAC-based expand (simple PRG). For higher-quality KDF use hkdf crate.
pub fn prg_hmac(seed: &[u8], out_len: usize) -> Vec<u8> {
    type H = Hmac<Sha256>;
    let mut out = Vec::new();
    let mut t = Vec::new();
    let mut counter = 1u8;
    while out.len() < out_len {
        let mut mac = H::new_from_slice(seed).unwrap();
        mac.update(&t);
        mac.update(&[counter]);
        let res = mac.finalize().into_bytes();
        t = res.to_vec();
        out.extend_from_slice(&t);
        counter = counter.wrapping_add(1);
    }
    out.truncate(out_len);
    out
}
