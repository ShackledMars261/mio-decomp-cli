mod constants;
mod decompiler;

use pyo3::prelude::*;

#[pymodule]
fn _mio_decomp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<decompiler::GinDecompiler>()?;
    Ok(())
}
