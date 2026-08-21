//! Standalone Quick Start program from the README.
//!
//! Run with: `cargo run --example quickstart`

use ternary_memory::{
    Decision, Episode, EpisodeKind, EpisodicMemory, LongTermMemory, MemoryConsolidation,
    ShortTermMemory,
};

fn main() {
    // Short-term memory: 100 slots with default Ebbinghaus decay.
    let mut stm = ShortTermMemory::with_capacity(100);

    // Store decisions.
    stm.store(Decision::new("explore_north", 0.7, 1).with_tag("frontier"));
    stm.store(Decision::new("attack_early", -0.3, 2).with_tag("combat"));
    stm.store(Decision::new("trade_silk", 0.9, 3).with_tag("economy"));

    // Long-term memory: per-label running statistics.
    let mut ltm = LongTermMemory::new();
    ltm.observe("explore_north", 0.7);
    ltm.observe("explore_north", 0.5);
    ltm.observe("explore_north", 0.8);
    // Later: retrieve statistics.
    if let Some(summary) = ltm.get("explore_north") {
        println!(
            "Mean: {:.2}, Std: {:.2}, n={}",
            summary.mean_outcome,
            summary.std_dev(),
            summary.count
        );
    }

    // Episodic memory: important events.
    let mut episodic = EpisodicMemory::new(1000);
    episodic.store(Episode::new(
        "Found gold mine!",
        EpisodeKind::Breakthrough,
        5,
        0.95,
    ));

    // Consolidate STM → LTM + Episodic.
    let consolidation = MemoryConsolidation::new();
    let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
    println!(
        "Consolidated {} entries, found {} episodes",
        result.consolidated_count, result.new_episodes
    );
}
