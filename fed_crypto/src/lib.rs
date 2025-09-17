use pyo3::prelude::*;

mod compressor;
// other modules (shamir/mask) can remain as before

#[pymodule]
fn fed_crypto(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_gradients_ext, m)?)?;
    Ok(())
}

/// compress_gradients_ext(arrays, compression_ratio) -> (values, indices, original_len, metadata_dict)
#[pyfunction]
fn compress_gradients_ext(py: Python, arrays: &PyAny, compression_ratio: f64) -> PyResult<(PyObject, PyObject, usize, PyObject)> {
    compressor::compress_gradients_ext(py, arrays, compression_ratio)
}
