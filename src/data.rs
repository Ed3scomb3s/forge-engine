//! Native Candle Data Aggregated Holder — Phase 5
//!
//! Stores OHLCV candle data natively in Rust, accepting numpy arrays once
//! at initialization. Provides binary-search-based range queries.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// RustCandleDataAggregated
// ---------------------------------------------------------------------------

#[pyclass]
pub struct RustCandleDataAggregated {
    /// Unix timestamps (seconds), sorted ascending.
    pub timestamps: Vec<i64>,
    /// OHLCV + metadata per candle: [open, high, low, close, volume,
    /// quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
    /// taker_buy_quote_asset_volume].
    pub values: Vec<[f64; 9]>,
}

#[pymethods]
impl RustCandleDataAggregated {
    /// Create from numpy arrays.
    ///
    /// - `timestamps_unix`: 1-D int64 array of Unix timestamps (seconds).
    /// - `values`: 2-D float64 array with shape (N, 9).
    #[new]
    fn new(
        timestamps_unix: PyReadonlyArray1<'_, i64>,
        values: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let ts_arr = timestamps_unix.as_array();
        let ts: Vec<i64> = ts_arr.iter().copied().collect();

        let vals_arr = values.as_array();
        let shape = vals_arr.shape();
        let nrows = shape[0];
        let ncols = shape[1];

        if ncols != 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected 9 columns in values array, got {}",
                ncols
            )));
        }

        if ts.len() != nrows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "timestamps length ({}) != values rows ({})",
                ts.len(),
                nrows
            )));
        }

        let mut vals = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let row = vals_arr.row(i);
            vals.push([
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
            ]);
        }

        Ok(Self {
            timestamps: ts,
            values: vals,
        })
    }

    /// Return (start_idx, end_idx) for the half-open range [start_unix, end_unix).
    ///
    /// Uses binary search (partition_point) for O(log n) performance.
    /// The returned indices can be used to slice both `timestamps` and `values`.
    #[pyo3(signature = (start_unix, end_unix))]
    fn find_range_indices(&self, start_unix: i64, end_unix: i64) -> (usize, usize) {
        let start_idx = self.timestamps.partition_point(|&ts| ts < start_unix);
        let end_idx = self.timestamps.partition_point(|&ts| ts < end_unix);
        (start_idx, end_idx)
    }

    /// Number of candles stored.
    fn __len__(&self) -> usize {
        self.timestamps.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn test_partition_point_logic() {
        // Simulate the binary search logic without numpy
        let timestamps: Vec<i64> = vec![100, 200, 300, 400, 500, 600];

        let start_unix = 200i64;
        let end_unix = 500i64;
        let start_idx = timestamps.partition_point(|&ts| ts < start_unix);
        let end_idx = timestamps.partition_point(|&ts| ts < end_unix);

        // [200, 300, 400] should be in range [200, 500)
        assert_eq!(start_idx, 1);
        assert_eq!(end_idx, 4);
        assert_eq!(&timestamps[start_idx..end_idx], &[200, 300, 400]);
    }

    #[test]
    fn test_partition_point_empty_range() {
        let timestamps: Vec<i64> = vec![100, 200, 300];
        let start_idx = timestamps.partition_point(|&ts| ts < 400);
        let end_idx = timestamps.partition_point(|&ts| ts < 400);
        assert_eq!(start_idx, end_idx);
        assert_eq!(start_idx, 3);
    }

    #[test]
    fn test_partition_point_full_range() {
        let timestamps: Vec<i64> = vec![100, 200, 300];
        let start_idx = timestamps.partition_point(|&ts| ts < 0);
        let end_idx = timestamps.partition_point(|&ts| ts < 1000);
        assert_eq!(start_idx, 0);
        assert_eq!(end_idx, 3);
    }
}
