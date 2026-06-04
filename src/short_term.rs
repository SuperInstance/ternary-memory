//! Short-term memory — fixed-size ring buffer of recent decisions with decay.

use crate::forgetting::ForgettingCurve;

/// A decision made by an agent, stored in short-term memory.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// A label identifying the action or choice.
    pub action: String,
    /// A scalar outcome score (higher = better).
    pub outcome: f64,
    /// Timestamp or tick when the decision was made.
    pub tick: u64,
    /// Context tags for indexing.
    pub tags: Vec<String>,
}

impl Decision {
    /// Create a new decision.
    pub fn new(action: impl Into<String>, outcome: f64, tick: u64) -> Self {
        Self {
            action: action.into(),
            outcome,
            tick,
            tags: Vec::new(),
        }
    }

    /// Add a context tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// A memory entry with computed retention based on age.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub decision: Decision,
    pub retention: f64,
}

/// Fixed-size ring buffer storing recent decisions with decay.
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
    pub fn new(capacity: usize, curve: ForgettingCurve) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        Self {
            buffer,
            capacity,
            head: 0,
            len: 0,
            curve,
            current_tick: 0,
        }
    }

    /// Create with default Ebbinghaus curve.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity, ForgettingCurve::default())
    }

    /// Store a decision. If the buffer is full, the oldest entry is overwritten.
    pub fn store(&mut self, decision: Decision) {
        self.current_tick = self.current_tick.max(decision.tick);
        self.buffer[self.head] = Some(decision);
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Return all entries with their current retention scores, newest first.
    pub fn recall(&self) -> Vec<MemoryEntry> {
        let mut entries = Vec::with_capacity(self.len);
        for i in 0..self.len {
            // Start from (head-1) and go backwards
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

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether memory is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }

    /// Return entries whose retention is above the given threshold.
    pub fn recall_above_threshold(&self, threshold: f64) -> Vec<MemoryEntry> {
        self.recall()
            .into_iter()
            .filter(|e| e.retention >= threshold)
            .collect()
    }

    /// Compute the average outcome of all retained entries (weighted by retention).
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

    /// Drain all entries, returning them. Resets the buffer.
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

    /// Get the current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
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
        let old_retention = entries.iter().find(|e| e.decision.action == "old").unwrap().retention;
        let recent_retention = entries.iter().find(|e| e.decision.action == "recent").unwrap().retention;
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
}
