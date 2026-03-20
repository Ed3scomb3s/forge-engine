//! Core trading types and exact Banker's Rounding — Phase 2
//!
//! All types mirror `forge_engine/trading.py` exactly.
//! Rounding uses 128-bit exact integer arithmetic to match Python's `round()`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Constants — must match forge_engine/trading.py exactly
// ─────────────────────────────────────────────────────────────────────────────
pub const OPEN_FEE_RATE: f64 = 0.0003;
pub const CLOSE_FEE_RATE: f64 = 0.0003;
pub const LIQ_FEE_RATE: f64 = 0.005;
pub const MMR: f64 = 0.005;
pub const MIN_NOTIONAL: f64 = 10.0;

// ─────────────────────────────────────────────────────────────────────────────
// Exact Banker's Rounding (CRITICAL)
//
// Python's `round(val, ndigits)` rounds the TRUE decimal value of the float,
// NOT the result of `x * 10^n` (which can introduce IEEE 754 rounding errors).
//
// Example: 2.675 is stored as 2.67499999999999982... in IEEE 754.
//   - Naive: 2.675 * 100 = 267.5 (multiplication rounds UP), → 268 → 2.68 WRONG
//   - Python: sees 2.67499..., rounds to 2.67 CORRECT
//
// Our approach: decompose the f64 into exact mantissa × 2^exp, then compute
// mantissa × 5^n × 2^(exp+n) using u128 integers (no precision loss for n≤22).
// This gives the exact product, and we apply Banker's rounding on that.
// ─────────────────────────────────────────────────────────────────────────────

/// Round an f64 to `ndigits` decimal places, matching Python's `round(x, ndigits)`.
///
/// Uses exact 128-bit integer arithmetic to avoid IEEE 754 intermediate rounding.
/// Valid for ndigits 0..=22 (covers all our use cases: 2 and 6).
pub fn python_round(x: f64, ndigits: u32) -> f64 {
    debug_assert!(ndigits <= 22);

    if !x.is_finite() || x == 0.0 {
        return x;
    }

    let sign = x.is_sign_negative();
    let x_abs = x.abs();

    // Decompose into mantissa and exponent: x_abs = mantissa * 2^exp
    let bits = x_abs.to_bits();
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;

    let (mantissa, exp) = if biased_exp == 0 {
        // Subnormal: no implicit leading 1, exponent is -1022
        let m = bits & 0x000F_FFFF_FFFF_FFFF;
        if m == 0 {
            return 0.0;
        }
        (m, -1022 - 52)
    } else {
        // Normal: implicit leading 1
        let m = (bits & 0x000F_FFFF_FFFF_FFFF) | (1u64 << 52);
        (m, biased_exp - 1023 - 52)
    };

    // Exact product: mantissa * 5^ndigits * 2^(exp + ndigits)
    //   because x * 10^n = mantissa * 2^exp * 2^n * 5^n
    //                     = mantissa * 5^n * 2^(exp+n)
    let pow5 = 5u128.pow(ndigits);
    let product = mantissa as u128 * pow5;
    let total_shift = exp + ndigits as i32;

    let rounded: u128 = if total_shift >= 0 {
        // Product is an integer — no rounding needed
        product << (total_shift as u32)
    } else {
        let shift = (-total_shift) as u32;
        if shift >= 128 {
            // Value is negligibly small
            return 0.0;
        }
        let integer_part = product >> shift;
        let remainder = product & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);

        if remainder > half {
            integer_part + 1
        } else if remainder < half {
            integer_part
        } else {
            // Exact tie: Banker's rounding (round to even)
            if integer_part & 1 == 1 {
                integer_part + 1
            } else {
                integer_part
            }
        }
    };

    // Convert back: result = rounded / 10^ndigits
    let pow10 = 10_f64.powi(ndigits as i32);
    let result = rounded as f64 / pow10;

    if sign { -result } else { result }
}

/// Round to 2 decimal places (price precision).
/// Matches Python's `round(x, 2)` exactly.
#[inline]
pub fn r_price(x: f64) -> f64 {
    python_round(x, 2)
}

/// Round to 6 decimal places (quantity precision).
/// Matches Python's `round(x, 6)` exactly.
#[inline]
pub fn r_qty(x: f64) -> f64 {
    python_round(x, 6)
}

/// Round to 6 decimal places (USD precision).
/// Matches Python's `round(x, 6)` exactly.
#[inline]
pub fn r_usd(x: f64) -> f64 {
    python_round(x, 6)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a UUID v4 as a lowercase hyphenated string.
#[inline]
pub fn uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Current UTC time as ISO 8601 string (matches Python `datetime.now(utc).isoformat()`).
#[inline]
pub fn now_utc_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Check if price is within [low, high] inclusive.
#[inline]
pub fn in_range(price: f64, low: f64, high: f64) -> bool {
    low <= price && price <= high
}

// ─────────────────────────────────────────────────────────────────────────────
// Enums — mirror Python str Enums; .value is the string representation
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    LONG,
    SHORT,
}

impl OrderSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LONG => "LONG",
            Self::SHORT => "SHORT",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "LONG" => Ok(Self::LONG),
            "SHORT" => Ok(Self::SHORT),
            _ => Err(format!("Invalid OrderSide: {}", s)),
        }
    }

    /// The opposite side (for closing positions).
    pub fn opposite(&self) -> Self {
        match self {
            Self::LONG => Self::SHORT,
            Self::SHORT => Self::LONG,
        }
    }
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PositionSide {
    LONG,
    SHORT,
}

impl PositionSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LONG => "LONG",
            Self::SHORT => "SHORT",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "LONG" => Ok(Self::LONG),
            "SHORT" => Ok(Self::SHORT),
            _ => Err(format!("Invalid PositionSide: {}", s)),
        }
    }
}

impl std::fmt::Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Convert between OrderSide and PositionSide (they share the same values).
impl From<OrderSide> for PositionSide {
    fn from(s: OrderSide) -> Self {
        match s {
            OrderSide::LONG => PositionSide::LONG,
            OrderSide::SHORT => PositionSide::SHORT,
        }
    }
}

impl From<PositionSide> for OrderSide {
    fn from(s: PositionSide) -> Self {
        match s {
            PositionSide::LONG => OrderSide::LONG,
            PositionSide::SHORT => OrderSide::SHORT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum OrderType {
    STOP_MARKET,
    CLOSE,
    TP,
    SL,
}

impl OrderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::STOP_MARKET => "STOP_MARKET",
            Self::CLOSE => "CLOSE",
            Self::TP => "TP",
            Self::SL => "SL",
        }
    }
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    OPEN,
    FILLED,
    CANCELLED,
    REJECTED,
}

impl OrderStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OPEN => "OPEN",
            Self::FILLED => "FILLED",
            Self::CANCELLED => "CANCELLED",
            Self::REJECTED => "REJECTED",
        }
    }
}

impl std::fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Order — mirrors Python Order dataclass
//
// `id` stored as String (UUID string) to avoid expensive FFI conversions.
// `created_at` stored as ISO 8601 String for the same reason.
// Python's to_dict outputs both as strings anyway.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: f64,
    pub quantity: f64,
    pub notional: f64,
    pub tp: Option<f64>,
    pub sl: Option<f64>,
    pub status: OrderStatus,
    pub created_at: String,
    pub eligible_from_index: Option<i64>,
    pub margin_pct: Option<f64>,
    pub allocated_margin: Option<f64>,
}

impl Order {
    /// Create a new Order with auto-generated UUID and current UTC timestamp.
    pub fn new(
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        price: f64,
        quantity: f64,
        notional: f64,
        tp: Option<f64>,
        sl: Option<f64>,
        eligible_from_index: Option<i64>,
        margin_pct: Option<f64>,
        allocated_margin: Option<f64>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            symbol,
            side,
            order_type,
            price,
            quantity,
            notional,
            tp,
            sl,
            status: OrderStatus::OPEN,
            created_at: now_utc_iso(),
            eligible_from_index,
            margin_pct,
            allocated_margin,
        }
    }

    /// Convert to a Python dict matching Python's `Order.to_dict()` exactly.
    ///
    /// Applies rounding: r_price to price/tp/sl, r_qty to quantity,
    /// r_usd to notional/allocated_margin/margin_pct.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("id", &self.id)?;
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", self.side.as_str())?;
        d.set_item("order_type", self.order_type.as_str())?;
        d.set_item("price", r_price(self.price))?;
        d.set_item("quantity", r_qty(self.quantity))?;
        d.set_item("notional", r_usd(self.notional))?;
        // tp: None -> Python None, Some(v) -> r_price(v)
        match self.tp {
            Some(v) => d.set_item("tp", r_price(v))?,
            None => d.set_item("tp", py.None())?,
        }
        // sl: None -> Python None, Some(v) -> r_price(v)
        match self.sl {
            Some(v) => d.set_item("sl", r_price(v))?,
            None => d.set_item("sl", py.None())?,
        }
        d.set_item("status", self.status.as_str())?;
        d.set_item("created_at", &self.created_at)?;
        // eligible_from_index: None -> Python None, Some(v) -> int
        match self.eligible_from_index {
            Some(v) => d.set_item("eligible_from_index", v)?,
            None => d.set_item("eligible_from_index", py.None())?,
        }
        // margin_pct: Python uses float(f"{x:.6f}") which is equiv to round(x, 6)
        match self.margin_pct {
            Some(v) => d.set_item("margin_pct", r_usd(v))?,
            None => d.set_item("margin_pct", py.None())?,
        }
        // allocated_margin: None -> Python None, Some(v) -> r_usd(v)
        match self.allocated_margin {
            Some(v) => d.set_item("allocated_margin", r_usd(v))?,
            None => d.set_item("allocated_margin", py.None())?,
        }
        Ok(d)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Position — mirrors Python Position dataclass
//
// Same string-based id/timestamp strategy as Order.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    pub side: PositionSide,
    pub entry_price: f64,
    pub size: f64,
    pub notional: f64,
    pub margin: f64,
    pub leverage: f64,
    pub liquidation_price: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub opened_at: String,
    pub opened_index: i64,
}

impl Position {
    /// Create a new Position with auto-generated UUID and current UTC timestamp.
    pub fn new(
        symbol: String,
        side: PositionSide,
        entry_price: f64,
        size: f64,
        notional: f64,
        margin: f64,
        leverage: f64,
        liquidation_price: f64,
        realized_pnl: f64,
        opened_index: i64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            symbol,
            side,
            entry_price,
            size,
            notional,
            margin,
            leverage,
            liquidation_price,
            realized_pnl,
            unrealized_pnl: 0.0,
            opened_at: now_utc_iso(),
            opened_index,
        }
    }

    /// Convert to a Python dict matching Python's `Position.to_dict(margin_mode, cross_liq_price)`.
    ///
    /// In isolated mode: liquidation_price = stored value (already rounded at calc time).
    /// In cross mode: liquidation_price = r_price(cross_liq_price) or None.
    pub fn to_dict<'py>(
        &self,
        py: Python<'py>,
        margin_mode: &str,
        cross_liq_price: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("id", &self.id)?;
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", self.side.as_str())?;
        d.set_item("entry_price", r_price(self.entry_price))?;
        d.set_item("size", r_qty(self.size))?;
        d.set_item("notional", r_usd(self.notional))?;
        d.set_item("margin", r_usd(self.margin))?;
        d.set_item("leverage", self.leverage)?;
        // liquidation_price: isolated uses stored value, cross uses cross_liq_price
        if margin_mode == "isolated" {
            d.set_item("liquidation_price", self.liquidation_price)?;
        } else {
            match cross_liq_price {
                Some(v) => d.set_item("liquidation_price", r_price(v))?,
                None => d.set_item("liquidation_price", py.None())?,
            }
        }
        d.set_item("realized_pnl", r_usd(self.realized_pnl))?;
        d.set_item("unrealized_pnl", r_usd(self.unrealized_pnl))?;
        d.set_item("opened_at", &self.opened_at)?;
        d.set_item("opened_index", self.opened_index)?;
        Ok(d)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — Prove Banker's rounding matches Python's round()
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── r_price (2 decimal places) ──────────────────────────────────────

    #[test]
    fn test_r_price_bankers_rounding_ties() {
        // 0.125 is exact in IEEE 754 (1/8).
        // 0.125 * 100 = 12.5 (exact tie) -> rounds to 12 (even).
        // Python: round(0.125, 2) == 0.12
        assert_eq!(r_price(0.125), 0.12);

        // 0.375 is exact in IEEE 754 (3/8).
        // 0.375 * 100 = 37.5 (exact tie) -> rounds to 38 (even).
        // Python: round(0.375, 2) == 0.38
        assert_eq!(r_price(0.375), 0.38);

        // 0.625 is exact (5/8). 62.5 -> 62 (even).
        // Python: round(0.625, 2) == 0.62
        assert_eq!(r_price(0.625), 0.62);

        // 0.875 is exact (7/8). 87.5 -> 88 (even).
        // Python: round(0.875, 2) == 0.88
        assert_eq!(r_price(0.875), 0.88);
    }

    #[test]
    fn test_r_price_float_representation_quirk() {
        // Classic Python gotcha: round(2.675, 2) == 2.67, NOT 2.68.
        // IEEE 754: 2.675 is stored as ~2.67499999999999982...
        // So it's below the midpoint and rounds DOWN.
        assert_eq!(r_price(2.675), 2.67);

        // Similarly: round(0.245, 2) == 0.24
        // 0.245 stored as ~0.244999999...
        assert_eq!(r_price(0.245), 0.24);
    }

    #[test]
    fn test_r_price_standard_cases() {
        assert_eq!(r_price(1.234), 1.23);
        assert_eq!(r_price(1.236), 1.24);
        assert_eq!(r_price(100.456), 100.46);
        assert_eq!(r_price(99999.999), 100000.0);
        assert_eq!(r_price(0.0), 0.0);
        assert_eq!(r_price(-1.235), -1.24); // negative: -123.5 -> -124 (even)
    }

    // ── r_qty / r_usd (6 decimal places) ────────────────────────────────

    #[test]
    fn test_r_qty_standard_cases() {
        // Clear non-tie cases (no float representation ambiguity)
        assert_eq!(r_qty(1.1234567), 1.123457);
        assert_eq!(r_qty(1.1234561), 1.123456);
        assert_eq!(r_qty(0.123456), 0.123456);
        assert_eq!(r_qty(0.0), 0.0);
    }

    #[test]
    fn test_r_usd_matches_r_qty() {
        // r_usd and r_qty both round to 6 decimals
        let values = [0.0, 1.0, 0.123456, 1.1234567, 99999.999999];
        for v in values {
            assert_eq!(r_usd(v), r_qty(v), "r_usd and r_qty diverge for {}", v);
        }
    }

    // ── round_ties_even directly (0-decimal equivalence) ────────────────

    #[test]
    fn test_round_ties_even_zero_decimals() {
        // Python: round(2.5) == 2 (half-to-even, 2 is even)
        assert_eq!((2.5_f64).round_ties_even(), 2.0);
        // Python: round(3.5) == 4 (half-to-even, 4 is even)
        assert_eq!((3.5_f64).round_ties_even(), 4.0);
        // Python: round(4.5) == 4
        assert_eq!((4.5_f64).round_ties_even(), 4.0);
        // Python: round(5.5) == 6
        assert_eq!((5.5_f64).round_ties_even(), 6.0);
        // Python: round(0.5) == 0
        assert_eq!((0.5_f64).round_ties_even(), 0.0);
        // Python: round(1.5) == 2
        assert_eq!((1.5_f64).round_ties_even(), 2.0);
    }

    // ── Enum conversions ────────────────────────────────────────────────

    #[test]
    fn test_enum_string_roundtrip() {
        assert_eq!(OrderSide::from_str("LONG").unwrap(), OrderSide::LONG);
        assert_eq!(OrderSide::from_str("SHORT").unwrap(), OrderSide::SHORT);
        assert!(OrderSide::from_str("INVALID").is_err());

        assert_eq!(OrderSide::LONG.as_str(), "LONG");
        assert_eq!(OrderSide::SHORT.as_str(), "SHORT");

        assert_eq!(PositionSide::from_str("LONG").unwrap(), PositionSide::LONG);
        assert_eq!(PositionSide::LONG.as_str(), "LONG");
    }

    #[test]
    fn test_side_conversions() {
        assert_eq!(PositionSide::from(OrderSide::LONG), PositionSide::LONG);
        assert_eq!(PositionSide::from(OrderSide::SHORT), PositionSide::SHORT);
        assert_eq!(OrderSide::from(PositionSide::LONG), OrderSide::LONG);
        assert_eq!(OrderSide::from(PositionSide::SHORT), OrderSide::SHORT);
        assert_eq!(OrderSide::LONG.opposite(), OrderSide::SHORT);
        assert_eq!(OrderSide::SHORT.opposite(), OrderSide::LONG);
    }

    // ── Constants ───────────────────────────────────────────────────────

    #[test]
    fn test_constants_match_python() {
        assert_eq!(OPEN_FEE_RATE, 0.0003);
        assert_eq!(CLOSE_FEE_RATE, 0.0003);
        assert_eq!(LIQ_FEE_RATE, 0.005);
        assert_eq!(MMR, 0.005);
        assert_eq!(MIN_NOTIONAL, 10.0);
    }
}
