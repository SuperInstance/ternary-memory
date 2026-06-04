//! Memory consolidation — periodically consolidate short-term → long-term.

use crate::short_term::{ShortTermMemory, Decision};
use crate::long_term::LongTermMemory;
use crate::episodic::{EpisodicMemory, Episode, EpisodeKind};

/// Result of a consolidation pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationResult {
    /// Number of short-term memories consolidated.
    pub consolidated_count: usize,
    /// Number of new episodes detected.
    pub new_episodes: usize,
    /// Labels that were updated in long-term memory.
    pub updated_labels: Vec<String>,
}

/// Configuration for what counts as an episode-worthy event.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Outcomes above this threshold are breakthroughs.
    pub breakthrough_threshold: f64,
    /// Outcomes below this threshold are near-misses.
    pub near_miss_threshold: f64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            breakthrough_threshold: 0.8,
            near_miss_threshold: -0.5,
        }
    }
}

/// Handles consolidation of short-term memory into long-term and episodic memory.
#[derive(Debug)]
pub struct MemoryConsolidation {
    config: ConsolidationConfig,
}

impl MemoryConsolidation {
    /// Create with default config.
    pub fn new() -> Self {
        Self { config: ConsolidationConfig::default() }
    }

    /// Create with custom config.
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    /// Consolidate short-term memory into long-term and episodic memory.
    /// Drains short-term memory and processes all entries.
    pub fn consolidate(
        &self,
        stm: &mut ShortTermMemory,
        ltm: &mut LongTermMemory,
        episodic: &mut EpisodicMemory,
    ) -> ConsolidationResult {
        let decisions = stm.drain();
        let count = decisions.len();
        let mut new_episodes = 0;
        let mut updated_labels = Vec::new();

        for decision in &decisions {
            // Update long-term memory
            ltm.observe(&decision.action, decision.outcome);
            if !updated_labels.contains(&decision.action) {
                updated_labels.push(decision.action.clone());
            }

            // Check for episodes
            if decision.outcome >= self.config.breakthrough_threshold {
                episodic.store(Episode::new(
                    format!("Breakthrough: {} (outcome={:.2})", decision.action, decision.outcome),
                    EpisodeKind::Breakthrough,
                    decision.tick,
                    decision.outcome,
                ));
                new_episodes += 1;
            } else if decision.outcome <= self.config.near_miss_threshold {
                episodic.store(Episode::new(
                    format!("Near miss: {} (outcome={:.2})", decision.action, decision.outcome),
                    EpisodeKind::NearMiss,
                    decision.tick,
                    decision.outcome,
                ));
                new_episodes += 1;
            }
        }

        ConsolidationResult {
            consolidated_count: count,
            new_episodes,
            updated_labels,
        }
    }

    /// Selectively consolidate only entries matching a predicate.
    pub fn consolidate_selective(
        &self,
        stm: &mut ShortTermMemory,
        ltm: &mut LongTermMemory,
        episodic: &mut EpisodicMemory,
        predicate: impl Fn(&Decision) -> bool,
    ) -> ConsolidationResult {
        let all = stm.drain();
        let mut keep = Vec::new();
        let mut consolidated = Vec::new();
        let mut new_episodes = 0;
        let mut updated_labels = Vec::new();

        for d in all {
            if predicate(&d) {
                consolidated.push(d);
            } else {
                keep.push(d);
            }
        }

        // Put back the ones we didn't consolidate
        for d in &keep {
            stm.store((*d).clone());
        }

        // Process consolidated
        for decision in &consolidated {
            ltm.observe(&decision.action, decision.outcome);
            if !updated_labels.contains(&decision.action) {
                updated_labels.push(decision.action.clone());
            }
            if decision.outcome >= self.config.breakthrough_threshold {
                episodic.store(Episode::new(
                    format!("Breakthrough: {} (outcome={:.2})", decision.action, decision.outcome),
                    EpisodeKind::Breakthrough,
                    decision.tick,
                    decision.outcome,
                ));
                new_episodes += 1;
            } else if decision.outcome <= self.config.near_miss_threshold {
                episodic.store(Episode::new(
                    format!("Near miss: {} (outcome={:.2})", decision.action, decision.outcome),
                    EpisodeKind::NearMiss,
                    decision.tick,
                    decision.outcome,
                ));
                new_episodes += 1;
            }
        }

        ConsolidationResult {
            consolidated_count: consolidated.len(),
            new_episodes,
            updated_labels,
        }
    }
}

impl Default for MemoryConsolidation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forgetting::ForgettingCurve;

    #[test]
    fn test_basic_consolidation() {
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(10, curve);
        let mut ltm = LongTermMemory::new();
        let mut episodic = EpisodicMemory::unlimited();
        let consolidation = MemoryConsolidation::new();

        stm.store(Decision::new("explore", 0.3, 1));
        stm.store(Decision::new("attack", 0.9, 2));
        stm.store(Decision::new("flee", -0.8, 3));

        let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
        assert_eq!(result.consolidated_count, 3);
        assert_eq!(result.new_episodes, 2); // attack=breakthrough, flee=near-miss
        assert!(stm.is_empty());
        assert_eq!(ltm.len(), 3);
    }

    #[test]
    fn test_selective_consolidation() {
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(10, curve);
        let mut ltm = LongTermMemory::new();
        let mut episodic = EpisodicMemory::unlimited();
        let consolidation = MemoryConsolidation::new();

        stm.store(Decision::new("explore", 0.3, 1));
        stm.store(Decision::new("attack", 0.9, 2));

        let result = consolidation.consolidate_selective(
            &mut stm, &mut ltm, &mut episodic,
            |d| d.action == "attack",
        );
        assert_eq!(result.consolidated_count, 1);
        assert_eq!(stm.len(), 1); // "explore" kept
    }

    #[test]
    fn test_consolidation_updates_labels() {
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(10, curve);
        let mut ltm = LongTermMemory::new();
        let mut episodic = EpisodicMemory::unlimited();
        let consolidation = MemoryConsolidation::new();

        stm.store(Decision::new("a", 0.5, 1));
        stm.store(Decision::new("b", 0.6, 2));
        stm.store(Decision::new("a", 0.4, 3));

        let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
        assert!(result.updated_labels.contains(&"a".to_string()));
        assert!(result.updated_labels.contains(&"b".to_string()));
        // "a" should appear only once
        assert_eq!(result.updated_labels.iter().filter(|l| *l == "a").count(), 1);
    }
}
