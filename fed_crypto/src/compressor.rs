// fed_crypto/src/compressor.rs
//
// Enhanced Rust compressor for AdaptiveGradientCompressor
// Exposes compress_gradients_ext(py, arrays, compression_ratio) -> (values_np, indices_np, original_len, metadata_dict)
//
// metadata_dict contains:
//   - "compression_ratio": float
//   - "k": int (number of kept elements)
//   - "original_len": int (total flattened length)
//   - "original_shapes": list of tuples, each tuple is the shape of the corresponding input array
//   - "dtype": "float32"
//
// No magic numbers are embedded; compression_ratio is provided by the caller.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::types::PySequence;
use std::cmp::Ordering;

#[pyfunction]
pub fn compress_gradients_ext(py: Python, arrays: &PyAny, compression_ratio: f64) -> PyResult<(PyObject, PyObject, usize, PyObject)> {
    if compression_ratio <= 0.0 {
        return Err(PyTypeError::new_err("compression_ratio must be > 0.0"));
    }

    let mut flat: Vec<f32> = Vec::new();
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    // Accept sequence or single numpy array
    if let Ok(seq) = arrays.extract::<&PySequence>() {
        let len = seq.len().unwrap_or(0);
        for i in 0..len {
            let item = seq.get_item(i)?;
            match PyReadonlyArrayDyn::<f32>::try_from(item) {
                Ok(arr) => {
                    let view = arr.as_array();
                    shapes.push(view.shape().iter().map(|x| *x as usize).collect());
                    for v in view.iter() {
                        flat.push(*v as f32);
                    }
                }
                Err(_) => {
                    // try float64 array then cast
                    match PyReadonlyArrayDyn::<f64>::try_from(item) {
                        Ok(arr64) => {
                            let view = arr64.as_array();
                            shapes.push(view.shape().iter().map(|x| *x as usize).collect());
                            for v in view.iter() {
                                flat.push(*v as f32);
                            }
                        }
                        Err(_) => {
                            return Err(PyTypeError::new_err(format!(
                                "Unsupported element type in sequence at index {}. Expected numpy array float32/float64.",
                                i
                            )));
                        }
                    }
                }
            }
        }
    } else {
        // single array
        if let Ok(arr) = PyReadonlyArrayDyn::<f32>::try_from(arrays) {
            let view = arr.as_array();
            shapes.push(view.shape().iter().map(|x| *x as usize).collect());
            for v in view.iter() {
                flat.push(*v as f32);
            }
        } else if let Ok(arr64) = PyReadonlyArrayDyn::<f64>::try_from(arrays) {
            let view = arr64.as_array();
            shapes.push(view.shape().iter().map(|x| *x as usize).collect());
            for v in view.iter() {
                flat.push(*v as f32);
            }
        } else {
            return Err(PyTypeError::new_err("Unsupported input: expected sequence of numpy arrays or a numpy array"));
        }
    }

    let original_len = flat.len();
    if original_len == 0 {
        let empty_vals = Vec::<f32>::new().into_pyarray(py).to_object(py);
        let empty_idx = Vec::<i64>::new().into_pyarray(py).to_object(py);
        let metadata = PyDict::new(py);
        metadata.set_item("compression_ratio", compression_ratio)?;
        metadata.set_item("k", 0)?;
        metadata.set_item("original_len", 0)?;
        metadata.set_item("original_shapes", PyList::empty(py))?;
        metadata.set_item("dtype", "float32")?;
        return Ok((empty_vals, empty_idx, 0usize, metadata.to_object(py)));
    }

    // compute k
    let mut k = ((original_len as f64) * compression_ratio).ceil() as usize;
    if k < 1 {
        k = 1;
    }
    if k >= original_len {
        // return all entries
        let vals_py = flat.clone().into_pyarray(py).to_object(py);
        let idxs: Vec<i64> = (0..original_len).map(|i| i as i64).collect();
        let idx_py = idxs.into_pyarray(py).to_object(py);
        let metadata = PyDict::new(py);
        metadata.set_item("compression_ratio", compression_ratio)?;
        metadata.set_item("k", original_len)?;
        metadata.set_item("original_len", original_len)?;
        let shapes_py = shapes_to_pylist(py, &shapes)?;
        metadata.set_item("original_shapes", shapes_py)?;
        metadata.set_item("dtype", "float32")?;
        return Ok((vals_py, idx_py, original_len, metadata.to_object(py)));
    }

    // build pairs (abs_value, index)
    let mut pairs: Vec<(f32, usize)> = flat.iter().enumerate().map(|(i, v)| (v.abs(), i)).collect();

    let nth = pairs.len().saturating_sub(k);
    pairs.select_nth_unstable_by(nth, |a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let topk = &pairs[nth..];
    let mut indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();
    indices.sort_unstable();
    let values: Vec<f32> = indices.iter().map(|&i| flat[i]).collect();
    let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();

    let vals_py = values.into_pyarray(py).to_object(py);
    let idx_py = indices_i64.into_pyarray(py).to_object(py);

    let metadata = PyDict::new(py);
    metadata.set_item("compression_ratio", compression_ratio)?;
    metadata.set_item("k", k)?;
    metadata.set_item("original_len", original_len)?;
    let shapes_py = shapes_to_pylist(py, &shapes)?;
    metadata.set_item("original_shapes", shapes_py)?;
    metadata.set_item("dtype", "float32")?;

    Ok((vals_py, idx_py, original_len, metadata.to_object(py)))
}

fn shapes_to_pylist(py: Python, shapes: &Vec<Vec<usize>>) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for s in shapes.iter() {
        let tup = PyTuple::new(py, s.iter().map(|d| d.clone()));
        list.append(tup)?;
    }
    Ok(list.to_object(py))
}
