use num_bigint::{BigUint, RandBigInt};
use num_traits::{One, Zero};
use rand::rngs::OsRng;

pub fn split_secret(secret: &[u8], n: usize, t: usize) -> Vec<(u64, BigUint)> {
    let p = BigUint::parse_bytes(b"170141183460469231731687303715884105727", 10).unwrap();
    let secret_int = BigUint::from_bytes_be(secret);
    assert!(secret_int < p);
    let mut rng = OsRng;
    let mut coeffs: Vec<BigUint> = Vec::with_capacity(t);
    coeffs.push(secret_int.clone());
    for _ in 1..t {
        coeffs.push(rng.gen_biguint_below(&p));
    }
    let mut res = Vec::new();
    for i in 1..=n {
        let x = BigUint::from(i as u64);
        let mut acc = BigUint::zero();
        for a in coeffs.iter().rev() {
            acc = (acc * &x + a) % &p;
        }
        res.push((i as u64, acc));
    }
    res
}

pub fn reconstruct_secret_pairs(pairs: &Vec<(u64, BigUint)>, secret_len: usize) -> Result<Vec<u8>, String> {
    let p = BigUint::parse_bytes(b"170141183460469231731687303715884105727", 10).unwrap();
    if pairs.is_empty() {
        return Err("no pairs".to_string());
    }
    let xs: Vec<BigUint> = pairs.iter().map(|(x, _)| BigUint::from(*x)).collect();
    let ys: Vec<BigUint> = pairs.iter().map(|(_, y)| y.clone() % &p).collect();
    let mut total = BigUint::zero();
    for j in 0..xs.len() {
        let xj = &xs[j];
        let yj = &ys[j];
        let mut num = BigUint::one();
        let mut den = BigUint::one();
        for m in 0..xs.len() {
            if m == j { continue; }
            let xm = &xs[m];
            num = (num * (&p - xm)) % &p;
            let diff = (xj + &p - xm) % &p;
            den = (den * diff) % &p;
        }
        // modular inverse of den
        let inv_den = modinv::modinv_biguint(&den, &p).ok_or("no inverse")?;
        let lj0 = (num * inv_den) % &p;
        total = (total + (yj * lj0) % &p) % &p;
    }
    let bytes = total.to_bytes_be();
    // pad to secret_len
    let mut out = vec![0u8; secret_len];
    let start = secret_len.saturating_sub(bytes.len());
    out[start..].copy_from_slice(&bytes);
    Ok(out)
}
