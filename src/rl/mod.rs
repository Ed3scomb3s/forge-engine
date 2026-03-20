//! RL Fast-Path Module — Phase 6
//!
//! Actions, Observations, Rewards, and RlConfig for zero-FFI step_rl.

pub mod actions;
pub mod observations;
pub mod rewards;

pub use actions::RlActionSpace;
pub use observations::{ObsFeatureType, ObsHistory, ObservationSpace};
pub use rewards::RewardFn;

/// Configuration for the RL fast-path, stored inside RustTradingEngine.
pub struct RlConfig {
    pub action_space: RlActionSpace,
    pub observation_space: ObservationSpace,
    pub reward_fn: RewardFn,
    pub history: ObsHistory,
    pub step_count: i64,
    pub prev_equity: f64,
    /// Tracks whether a liquidation occurred in the most recent step.
    pub last_was_liquidation: bool,
}

impl RlConfig {
    pub fn reset(&mut self, initial_equity: f64) {
        self.observation_space.reset();
        self.reward_fn.reset();
        self.history.reset(initial_equity);
        self.step_count = 0;
        self.prev_equity = initial_equity;
        self.last_was_liquidation = false;
    }
}
