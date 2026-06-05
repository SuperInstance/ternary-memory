# ternary-memory

Memory systems for ternary agents — short-term, long-term, and episodic memory that agents use to make better decisions.

## Why Memory Matters for Agents

An agent without memory is condemned to repeat the same mistakes forever. Biological agents (animals, humans) have evolved sophisticated memory systems that allow them to:

1. **React quickly** to recent events (short-term memory)
2. **Generalize** from accumulated experience (long-term memory)
3. **Learn from pivotal moments** — near-misses, breakthroughs, surprises (episodic memory)

This crate implements these three memory subsystems for use in autonomous agents, game AI, reinforcement learning, and decision systems.

## Architecture

### ShortTermMemory

A fixed-size ring buffer of recent decisions. Each entry decays over time according to a configurable forgetting curve. When the buffer is full, the oldest entries are overwritten.

Use it for:
- Recent context for decision-making
- Weighted recency bias in action selection
- Temporary scratch space before consolidation

### LongTermMemory

A compressed summary of all past experience using running statistics (Welford's online algorithm). Tracks counts, means, variance, best/worst outcomes per action category.

Use it for:
- Quick policy decisions ("which action has the best historical outcome?")
- Statistical confidence estimates
- Merging experience from multiple agents

### EpisodicMemory

Stores specific important episodes — near-misses, breakthroughs, surprises. These are the moments an agent should never forget, regardless of how much time passes.

Use it for:
- Learning from critical events
- Storytelling and narrative generation
- Safety constraints ("never do X again — remember what happened last time")

### MemoryIndex

Generic context-tag-based indexing for fast retrieval of any memory type. Supports AND (all tags match) and OR (any tag match) queries with relevance sorting.

### ForgettingCurve

Configurable Ebbinghaus-style forgetting models:
- **Exponential decay** (Ebbinghaus): `R = e^(-t/S)`
- **Power-law decay**: `R = (1 + t)^(-α)`
- **Linear decay**: `R = 1 - t/H`

Each model computes a retention probability [0, 1] given an age `t`. Used by short-term memory to weight recency.

### MemoryConsolidation

The process of transferring knowledge from short-term to long-term memory, inspired by sleep consolidation in biological agents. During consolidation:
- Short-term entries are drained and processed into long-term summaries
- Exceptional outcomes are extracted as episodes
- Configurable thresholds control what counts as "exceptional"

## Usage

```rust
use ternary_memory::*;

// Set up memory systems
let curve = ForgettingCurve::ebbinghaus_with_half_life(50.0);
let mut stm = ShortTermMemory::new(100, curve);
let mut ltm = LongTermMemory::new();
let mut episodic = EpisodicMemory::new(50);

// Agent makes decisions
stm.store(Decision::new("explore", 0.3, 1));
stm.store(Decision::new("attack", 0.95, 2));

// Consolidate periodically
let consolidation = MemoryConsolidation::new();
let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
println!("Consolidated {} entries, found {} episodes", 
    result.consolidated_count, result.new_episodes);

// Use long-term memory for policy
if let Some(best) = ltm.best_label() {
    println!("Best action historically: {}", best);
}
```

## Design Principles

- **Pure Rust**: No unsafe code, no external dependencies
- **Composable**: Memory systems are independent; use what you need
- **Configurable**: Forgetting curves, capacities, and thresholds are all tunable
- **Efficient**: O(1) short-term store, O(n) consolidation, Welford's online stats

## See Also

- **ternary-agent** — Core agent types with ternary state
- **ternary-archive** — Archival and retrieval of ternary agent histories
- **ternary-database** — Persistent storage for ternary data
- **ternary-replay** — Replay and analysis of past agent decisions
- **ternary-chronicle** — Chronological event logging for ternary systems

## License

MIT
