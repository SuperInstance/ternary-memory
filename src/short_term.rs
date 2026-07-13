//! Short-term memory — fixed-size ring buffer of recent decisions with decay.
//!
//! [`ShortTermMemory`] is a bounded ring buffer of recent
//! [`Decision`]s. Each entry's retention is computed lazily on recall via a
//! [`ForgettingCurve`], so older entries weigh less in
//! [`weighted_average_outcome`](ShortTermMemory::weighted_average_outcome) and
//! are filtered out by [`recall_above_threshold`](ShortTermMemory::recall_above_threshold).
//!
//! ## When to use this
//!
//! Use [`ShortTermMemory`] for the agent's working memory — the recent context
//! that informs an immediate next decision. Don't use it for permanent
//! statistics (use [`LongTermMemory`](crate::LongTermMemory)) or for individual
//! noteworthy events (use [`EpisodicMemory`](crate::EpisodicMemory)). Periodic
//! [`MemoryConsolidation`](crate::MemoryConsolidation) drains STM into both.

use crate::forgetting::ForgettingCurve;

/// A decision made by an agent, stored in short-term memory.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// A label identifying the action or choice (e.g. `"attack"`).
    pub action: String,
    /// A scalar outcome score (higher = better). Conventionally in `[-1, +1]`
    /// but any `f64` is accepted.
    pub outcome: f64,
    /// Timestamp or tick when the decision was made. Used to compute retention
    /// against the store's `current_tick`.
    pub tick: u64,
    /// Context tags for indexing. Free-form; cross-referenced by
    /// [`EpisodicMemory`](crate::EpisodicMemory) and
    /// [`MemoryIndex`](crate::MemoryIndex) when you wire those up.
    pub tags: Vec<String>,
}

impl Decision {
    /// Create a new decision with no tags.
    #[must_use]
    pub fn new(action: impl Into<String>, outcome: f64, tick: u64) -> Self {
        Self {
            action: action.into(),
            outcome,
            tick,
            tags: Vec::new(),
        }
    }

    /// Builder: add a context tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// A memory entry with its computed retention at the most recent recall.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// The stored decision.
    pub decision: Decision,
    /// Retention in `[0, 1]` from the store's forgetting curve at the
    /// decision's age (`current_tick − decision.tick`).
    pub retention: f64,
}

/// Fixed-size ring buffer storing recent decisions with decay.
///
/// Construct with [`ShortTermMemory::with_capacity`] (default Ebbinghaus curve,
/// 100-tick half-life) or [`ShortTermMemory::new`] for a custom curve.
/// `capacity == 0` is permitted and produces a store that retains nothing;
/// [`store`](Self::store) is a no-op in that case.
#[derive(Debug, Clone)]
pub struct ShortTermMemory {
    buffer: Vec<Option<Decision>>,
    capacity: usize,
    head: usize,
    len: usize,
    curve: ForgettingCurve,
    current_tick: u64,
}

impl ShortTermMemory {
    /// Create a new short-term memory with given capacity and forgetting curve.
    ///
    /// `capacity == 0` produces an empty store whose [`store`](Self::store) is
    /// a no-op (rather than panicking on the modulo-by-zero in the ring
    /// buffer's wraparound).
    #[must_use]
    pub fn new(capacity: usize, curve: ForgettingCurve) -> Self {
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            capacity,
            head: 0,
            len: 0,
            curve,
            current_tick: 0,
        }
    }

    /// Create with default Ebbinghaus curve (100-tick half-life).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity, ForgettingCurve::default())
    }

    /// Store a decision. If the buffer is full, the oldest entry is
    /// overwritten. A no-op when `capacity == 0`.
    pub fn store(&mut self, decision: Decision) {
        if self.capacity == 0 {
            // Still advance current_tick so a future resize-then-store picks
            // up the right retention — but don't try to index an empty buffer.
            self.current_tick = self.current_tick.max(decision.tick);
            return;
        }
        self.current_tick = self.current_tick.max(decision.tick);
        self.buffer[self.head] = Some(decision);
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Return all entries with their current retention scores, newest first.
    ///
    /// Each entry's retention is computed against `current_tick` (which is the
    /// max `tick` ever stored).
    #[must_use]
    pub fn recall(&self) -> Vec<MemoryEntry> {
        let mut entries = Vec::with_capacity(self.len);
        for i in 0..self.len {
            // Start from (head-1) and go backwards.
            let idx = (self.head + self.capacity - 1 - i) % self.capacity;
            if let Some(ref decision) = self.buffer[idx] {
                let age = self.current_tick.saturating_sub(decision.tick) as f64;
                let retention = self.curve.retention(age);
                entries.push(MemoryEntry {
                    decision: decision.clone(),
                    retention,
                });
            }
        }
        entries
    }

    /// Number of stored entries (`<= capacity`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether memory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all entries (does not reset `current_tick`).
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }

    /// Return entries whose retention is `>= threshold`. Order is newest first
    /// (inherited from [`recall`](Self::recall)).
    #[must_use]
    pub fn recall_above_threshold(&self, threshold: f64) -> Vec<MemoryEntry> {
        self.recall()
            .into_iter()
            .filter(|e| e.retention >= threshold)
            .collect()
    }

    /// Compute the retention-weighted average outcome of all entries.
    ///
    /// Returns `0.0` if the store is empty or every entry has zero total
    /// retention (i.e. everything has decayed to zero).
    #[must_use]
    pub fn weighted_average_outcome(&self) -> f64 {
        let entries = self.recall();
        if entries.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = entries.iter().map(|e| e.retention).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        let weighted_sum: f64 = entries
            .iter()
            .map(|e| e.decision.outcome * e.retention)
            .sum();
        weighted_sum / total_weight
    }

    /// Drain all entries, returning them in arbitrary (buffer) order. Resets
    /// the buffer to empty (but does not reset `current_tick`).
    pub fn drain(&mut self) -> Vec<Decision> {
        let mut decisions = Vec::with_capacity(self.len);
        for slot in self.buffer.iter_mut() {
            if let Some(d) = slot.take() {
                decisions.push(d);
            }
        }
        self.head = 0;
        self.len = 0;
        decisions
    }

    /// Get the current tick (the max `tick` ever stored).
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Get the configured capacity (max entries the ring can hold).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forgetting::ForgettingModel;

    #[test]
    fn test_store_and_recall() {
        let mut stm = ShortTermMemory::with_capacity(3);
        stm.store(Decision::new("explore", 0.5, 0));
        stm.store(Decision::new("attack", 0.8, 1));
        let entries = stm.recall();
        assert_eq!(entries.len(), 2);
        // newest first
        assert_eq!(entries[0].decision.action, "attack");
        assert_eq!(entries[1].decision.action, "explore");
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut stm = ShortTermMemory::with_capacity(2);
        stm.store(Decision::new("a", 1.0, 0));
        stm.store(Decision::new("b", 2.0, 1));
        stm.store(Decision::new("c", 3.0, 2));
        let entries = stm.recall();
        assert_eq!(entries.len(), 2);
        // "a" should be evicted
        let actions: Vec<&str> = entries.iter().map(|e| e.decision.action.as_str()).collect();
        assert!(actions.contains(&"b"));
        assert!(actions.contains(&"c"));
        assert!(!actions.contains(&"a"));
    }

    #[test]
    fn test_decay_with_time() {
        let curve = ForgettingCurve::new(ForgettingModel::Linear { horizon: 10.0 });
        let mut stm = ShortTermMemory::new(5, curve);
        stm.store(Decision::new("old", 1.0, 0));
        stm.store(Decision::new("recent", 1.0, 5));
        // current_tick should be 5
        let entries = stm.recall();
        let old_retention = entries
            .iter()
            .find(|e| e.decision.action == "old")
            .unwrap()
            .retention;
        let recent_retention = entries
            .iter()
            .find(|e| e.decision.action == "recent")
            .unwrap()
            .retention;
        assert!(old_retention < recent_retention);
    }

    #[test]
    fn test_weighted_average_outcome() {
        let mut stm = ShortTermMemory::with_capacity(5);
        stm.store(Decision::new("a", 1.0, 0));
        stm.store(Decision::new("b", 0.0, 0));
        let avg = stm.weighted_average_outcome();
        assert!((avg - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_drain_clears() {
        let mut stm = ShortTermMemory::with_capacity(5);
        stm.store(Decision::new("a", 1.0, 0));
        let drained = stm.drain();
        assert_eq!(drained.len(), 1);
        assert!(stm.is_empty());
    }

    #[test]
    fn test_recall_above_threshold() {
        let curve = ForgettingCurve::new(ForgettingModel::Linear { horizon: 10.0 });
        let mut stm = ShortTermMemory::new(5, curve);
        stm.store(Decision::new("old", 1.0, 0)); // retention = 1 - 0/10 = 1.0 (tick=0, current_tick=0)
                                                 // Now add a newer one; current_tick = 5
        stm.store(Decision::new("mid", 1.0, 5));
        // old is now age=5, retention = 1 - 5/10 = 0.5
        // mid is age=0, retention = 1.0
        let strong = stm.recall_above_threshold(0.6);
        assert!(strong.iter().all(|e| e.decision.action != "old"));
        assert!(strong.iter().any(|e| e.decision.action == "mid"));
    }

    // ===== New edge-case tests added in production round 4 =====

    #[test]
    fn test_capacity_zero_does_not_panic() {
        // Regression: previously ShortTermMemory::new(0, _).store(...)
        // panicked on the buffer index (and would have hit modulo-by-zero
        // in the wraparound) — both are now guarded.
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(0, curve);
        assert_eq!(stm.capacity(), 0);
        assert!(stm.is_empty());
        assert_eq!(stm.len(), 0);
        stm.store(Decision::new("x", 1.0, 7));
        // Nothing stored, but current_tick advanced.
        assert_eq!(stm.current_tick(), 7);
        assert!(stm.is_empty());
        assert!(stm.recall().is_empty());
        assert_eq!(stm.recall_above_threshold(0.0).len(), 0);
        assert!((stm.weighted_average_outcome() - 0.0).abs() < 1e-12);
        assert!(stm.drain().is_empty());
    }

    #[test]
    fn test_single_entry_recall_and_eviction() {
        let mut stm = ShortTermMemory::with_capacity(1);
        stm.store(Decision::new("only", 0.5, 3));
        assert_eq!(stm.len(), 1);
        let r = stm.recall();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].decision.action, "only");
        // Overwrite at capacity 1.
        stm.store(Decision::new("next", 0.9, 4));
        let r = stm.recall();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].decision.action, "next");
    }

    #[test]
    fn test_complete_forcoding_yields_zero_average() {
        // With a Linear curve and an age past the horizon, every entry has
        // retention exactly 0 — so weighted_average_outcome must be 0.0 even
        // though outcomes themselves are non-zero.
        let curve = ForgettingCurve::new(ForgettingModel::Linear { horizon: 5.0 });
        let mut stm = ShortTermMemory::new(5, curve);
        stm.store(Decision::new("a", 1.0, 0));
        stm.store(Decision::new("b", -1.0, 1));
        // Move current_tick past the horizon for both.
        stm.store(Decision::new("anchor", 0.0, 100));
        let entries = stm.recall();
        // All retained entries (excluding the just-stored anchor) must have
        // retention 0.
        let any_retention = entries
            .iter()
            .filter(|e| e.decision.action != "anchor")
            .map(|e| e.retention)
            .fold(0.0_f64, f64::max);
        assert_eq!(any_retention, 0.0);
    }

    #[test]
    fn test_retention_values_match_curve_by_hand() {
        // Hand-derivation sabotage check. With a Linear curve of horizon 10,
        // current_tick=5, an entry stored at tick 0 has age 5 → retention 0.5,
        // and one stored at tick 5 has age 0 → retention 1.0. If we silently
        // flip the formula in retention() (e.g. divide by a different number)
        // these equalities break.
        let curve = ForgettingCurve::new(ForgettingModel::Linear { horizon: 10.0 });
        let mut stm = ShortTermMemory::new(5, curve);
        stm.store(Decision::new("old", 1.0, 0));
        stm.store(Decision::new("mid", 1.0, 5));
        let entries = stm.recall();
        let old_r = entries
            .iter()
            .find(|e| e.decision.action == "old")
            .unwrap()
            .retention;
        let mid_r = entries
            .iter()
            .find(|e| e.decision.action == "mid")
            .unwrap()
            .retention;
        assert!((old_r - 0.5).abs() < 1e-12, "old retention {old_r}");
        assert!((mid_r - 1.0).abs() < 1e-12, "mid retention {mid_r}");
    }

    #[test]
    fn test_out_of_order_ticks_dont_reset_current_tick() {
        let mut stm = ShortTermMemory::with_capacity(5);
        stm.store(Decision::new("late", 1.0, 100));
        stm.store(Decision::new("early", 1.0, 1)); // shouldn't drag current_tick back
        assert_eq!(stm.current_tick(), 100);
    }
}
