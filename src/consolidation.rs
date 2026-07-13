//! Memory consolidation — periodically consolidate short-term → long-term.
//!
//! [`MemoryConsolidation`] drains [`ShortTermMemory`](crate::ShortTermMemory)
//! into both [`LongTermMemory`](crate::LongTermMemory) (every drained decision
//! becomes a labelled observation) and [`EpisodicMemory`] (decisions whose
//! outcome is salient enough — above `breakthrough_threshold` or at/below
//! `near_miss_threshold` — are stored as episodes).
//!
//! ## When to use this
//!
//! Call [`consolidate`](MemoryConsolidation::consolidate) periodically — for
//! example, between agent episodes, at the end of a "day", or when STM is
//! nearing capacity. This implements the sleep-consolidation hypothesis
//! (Diekelmann & Born, 2010): short-term memories are transferred to long-term
//! storage during quiescent periods. Use
//! [`consolidate_selective`](MemoryConsolidation::consolidate_selective) when
//! you want to keep some decisions in STM (e.g. only consolidate decisions the
//! agent has finished reasoning about).

use crate::episodic::{Episode, EpisodeKind, EpisodicMemory};
use crate::long_term::LongTermMemory;
use crate::short_term::{Decision, ShortTermMemory};

/// Result of a consolidation pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationResult {
    /// Number of short-term memories consolidated.
    pub consolidated_count: usize,
    /// Number of new episodes detected.
    pub new_episodes: usize,
    /// Distinct labels that were updated in long-term memory, in first-seen
    /// order.
    pub updated_labels: Vec<String>,
}

/// Configuration for what counts as an episode-worthy event.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationConfig {
    /// Outcomes `>=` this threshold are stored as [`EpisodeKind::Breakthrough`].
    pub breakthrough_threshold: f64,
    /// Outcomes `<=` this threshold are stored as [`EpisodeKind::NearMiss`].
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
///
/// Construct with [`MemoryConsolidation::new`] (default config) or
/// [`MemoryConsolidation::with_config`] for custom thresholds.
#[derive(Debug)]
pub struct MemoryConsolidation {
    config: ConsolidationConfig,
}

impl MemoryConsolidation {
    /// Create with default config (breakthrough ≥ 0.8, near-miss ≤ −0.5).
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ConsolidationConfig::default(),
        }
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    /// Consolidate short-term memory into long-term and episodic memory.
    ///
    /// Drains short-term memory completely and processes every entry. For each
    /// drained decision:
    ///
    /// 1. Observes `(action, outcome)` in long-term memory.
    /// 2. If `outcome >= breakthrough_threshold`, stores a
    ///    [`EpisodeKind::Breakthrough`] episode.
    /// 3. Otherwise, if `outcome <= near_miss_threshold`, stores a
    ///    [`EpisodeKind::NearMiss`] episode.
    ///
    /// Breakthrough and near-miss are mutually exclusive — a single decision
    /// produces at most one episode.
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
            self.observe_and_maybe_episode(
                decision,
                ltm,
                episodic,
                &mut new_episodes,
                &mut updated_labels,
            );
        }

        ConsolidationResult {
            consolidated_count: count,
            new_episodes,
            updated_labels,
        }
    }

    /// Selectively consolidate only entries matching a predicate.
    ///
    /// Same semantics as [`consolidate`](Self::consolidate) but entries for
    /// which `predicate` returns `false` are put back into STM (preserving
    /// their original `tick`). The returned `ConsolidationResult` describes
    /// only the consolidated subset.
    pub fn consolidate_selective(
        &self,
        stm: &mut ShortTermMemory,
        ltm: &mut LongTermMemory,
        episodic: &mut EpisodicMemory,
        predicate: impl Fn(&Decision) -> bool,
    ) -> ConsolidationResult {
        let all = stm.drain();
        let mut consolidated = 0usize;
        let mut new_episodes = 0;
        let mut updated_labels = Vec::new();

        for d in all {
            if predicate(&d) {
                consolidated += 1;
                self.observe_and_maybe_episode(
                    &d,
                    ltm,
                    episodic,
                    &mut new_episodes,
                    &mut updated_labels,
                );
            } else {
                stm.store(d);
            }
        }

        ConsolidationResult {
            consolidated_count: consolidated,
            new_episodes,
            updated_labels,
        }
    }

    fn observe_and_maybe_episode(
        &self,
        decision: &Decision,
        ltm: &mut LongTermMemory,
        episodic: &mut EpisodicMemory,
        new_episodes: &mut usize,
        updated_labels: &mut Vec<String>,
    ) {
        ltm.observe(&decision.action, decision.outcome);
        if !updated_labels.contains(&decision.action) {
            updated_labels.push(decision.action.clone());
        }

        if decision.outcome >= self.config.breakthrough_threshold {
            episodic.store(Episode::new(
                format!(
                    "Breakthrough: {} (outcome={:.2})",
                    decision.action, decision.outcome
                ),
                EpisodeKind::Breakthrough,
                decision.tick,
                decision.outcome,
            ));
            *new_episodes += 1;
        } else if decision.outcome <= self.config.near_miss_threshold {
            episodic.store(Episode::new(
                format!(
                    "Near miss: {} (outcome={:.2})",
                    decision.action, decision.outcome
                ),
                EpisodeKind::NearMiss,
                decision.tick,
                decision.outcome,
            ));
            *new_episodes += 1;
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

        let result = consolidation
            .consolidate_selective(&mut stm, &mut ltm, &mut episodic, |d| d.action == "attack");
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
        assert_eq!(
            result.updated_labels.iter().filter(|l| *l == "a").count(),
            1
        );
    }

    #[test]
    fn test_consolidate_hand_traced() {
        // Hand-derivation check of the STM → LTM transfer + episode detection.
        //
        // Setup: store three decisions with outcomes {0.3, 0.9, -0.8}.
        // Default config: breakthrough ≥ 0.8, near-miss ≤ -0.5.
        //
        // After consolidation:
        //   - LTM should have three labels with the same mean as the single
        //     observation (one obs per label).
        //   - attack (0.9) → Breakthrough episode.
        //   - flee (-0.8)  → NearMiss episode.
        //   - explore (0.3) → no episode (between thresholds).
        //   - mutually-exclusive branch: a single decision never produces both.
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(10, curve);
        let mut ltm = LongTermMemory::new();
        let mut episodic = EpisodicMemory::unlimited();
        let consolidation = MemoryConsolidation::new();

        stm.store(Decision::new("explore", 0.3, 1));
        stm.store(Decision::new("attack", 0.9, 2));
        stm.store(Decision::new("flee", -0.8, 3));

        let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);

        // Per-label mean equals the single observation.
        assert!((ltm.get("explore").unwrap().mean_outcome - 0.3).abs() < 1e-12);
        assert!((ltm.get("attack").unwrap().mean_outcome - 0.9).abs() < 1e-12);
        assert!((ltm.get("flee").unwrap().mean_outcome - (-0.8)).abs() < 1e-12);

        // Episode detection matches the hand calculation.
        assert_eq!(result.new_episodes, 2);
        assert_eq!(episodic.recall_by_kind(&EpisodeKind::Breakthrough).len(), 1);
        assert_eq!(episodic.recall_by_kind(&EpisodeKind::NearMiss).len(), 1);
        assert_eq!(episodic.len(), 2);
    }

    #[test]
    fn test_consolidate_empty_stm_is_noop() {
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(10, curve);
        let mut ltm = LongTermMemory::new();
        let mut episodic = EpisodicMemory::unlimited();
        let consolidation = MemoryConsolidation::new();

        let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
        assert_eq!(result.consolidated_count, 0);
        assert_eq!(result.new_episodes, 0);
        assert!(result.updated_labels.is_empty());
        assert!(stm.is_empty());
        assert!(ltm.is_empty());
        assert!(episodic.is_empty());
    }

    #[test]
    fn test_thresholds_are_mutually_exclusive() {
        // Outcome exactly at breakthrough and exactly at near-miss: with the
        // documented semantics each decision produces at most one episode
        // (else-if branch). Verify by storing both and counting.
        let curve = ForgettingCurve::default();
        let mut stm = ShortTermMemory::new(10, curve);
        let mut ltm = LongTermMemory::new();
        let mut episodic = EpisodicMemory::unlimited();
        let consolidation = MemoryConsolidation::new();

        stm.store(Decision::new("hi", 0.8, 1)); // exactly breakthrough
        stm.store(Decision::new("lo", -0.5, 2)); // exactly near-miss

        let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
        assert_eq!(result.consolidated_count, 2);
        assert_eq!(result.new_episodes, 2);
        assert_eq!(episodic.len(), 2);
    }
}
