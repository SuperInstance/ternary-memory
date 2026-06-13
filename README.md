# ternary-memory

Multi-tier memory systems for ternary agents. Implements short-term (ring buffer with Ebbinghaus decay), long-term (Welford running statistics), and episodic memory (salience-filtered event store) with context-tag indexing and periodic consolidation — the cognitive architecture for agents that learn from experience.

## Why It Matters

Agents without memory repeat the same mistakes. Agents with only recent memory can't recognize long-term patterns. This crate provides the **three complementary memory systems** identified by cognitive science (Tulving, 1972; Atkinson & Shiffrin, 1968):

| System | Capacity | Duration | Function |
|--------|----------|----------|----------|
| Short-term (STM) | Bounded ring | Minutes | Recent context for immediate decisions |
| Long-term (LTM) | Unbounded summaries | Permanent | Accumulated statistics (mean, variance, extremes) |
| Episodic | Bounded event log | Permanent | Specific important events (breakthroughs, near-misses) |

The forgetting curve is configurable: Ebbinghaus exponential, power-law, or linear decay — modeling different memory retention profiles observed in cognitive psychology.

## How It Works

### Short-Term Memory: Ring Buffer with Decay

STM is a fixed-capacity ring buffer. When full, the oldest entry is overwritten. Each entry's retention is computed from the **Ebbinghaus forgetting curve**:

```
R(t) = e^(−t / S)
```

where t = elapsed time since storage, S = stability parameter (half-life analog). At t = S · ln(2) ≈ 0.693·S, retention drops to 50%.

**Three forgetting models**:

| Model | Formula | Use case |
|-------|---------|----------|
| Ebbinghaus | R = e^(−t/S) | Biological memory (default) |
| Power-law | R = (1+t)^(−α) | Skill/knowledge retention |
| Linear | R = 1 − t/H | Hard-deadline expiry |

### Long-Term Memory: Welford's Online Algorithm

LTM maintains running statistics without storing individual observations. **Welford's algorithm** (Welford, 1962) provides numerically stable single-pass variance:

```
count ← count + 1
δ ← x − mean
mean ← mean + δ / count
δ₂ ← x − mean
M₂ ← M₂ + δ · δ₂

variance = M₂ / count        (population)
variance = M₂ / (count − 1)  (sample, Bessel-corrected)
```

This avoids the catastrophic cancellation that affects the naive two-pass formula Σx² − (Σx)²/n.

**Confidence** measure:

```
C(n) = 1 − 1/(1 + √n)
```

At n = 1, C = 0.29; n = 10, C = 0.76; n = 100, C = 0.91; n → ∞, C → 1.

### Episodic Memory: Salience-Filtered Events

Episodic memory stores only **noteworthy** events, filtered by configurable thresholds:

- **Breakthrough**: outcome ≥ breakthrough_threshold (default: 0.8)
- **Near-miss**: outcome ≤ near_miss_threshold (default: −0.5)
- **Surprise**: unexpected outcome (user-defined criterion)

When capacity is reached, the oldest episode is evicted (FIFO). This bounded design ensures episodic memory never causes unbounded memory growth.

### Memory Index: Tag-Based Retrieval

All memory entries can be tagged with context keys for fast retrieval:

```
ContextTag = { key: String, value: String }
```

Two query modes:
- **query_all(tags)**: AND semantics — match all specified tags
- **query_any(tags)**: OR semantics — match any specified tag

Results are sorted by relevance (descending), enabling priority-weighted retrieval.

### Memory Consolidation

Periodic consolidation transfers STM entries to LTM statistics:

```
For each decision in STM.drain():
    LTM.observe(decision.action, decision.outcome)
    If decision.outcome ≥ breakthrough_threshold:
        Episodic.store(Breakthrough episode)
    If decision.outcome ≤ near_miss_threshold:
        Episodic.store(NearMiss episode)
```

This implements the **sleep consolidation** hypothesis (Diekelmann & Born, 2010): short-term memories are transferred to long-term storage during quiescent periods.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| STM::store(d) | O(1) | O(1) |
| STM::recent(n) | O(n) | O(n) |
| STM::drain() | O(capacity) | O(capacity) |
| LTM::observe(label, x) | O(1) | O(1) |
| LTM::summary(label) | O(k) | O(1) |
| Episodic::store(e) | O(1) | O(1) |
| Episodic::query(criterion) | O(n) | O(k) |
| Index::query_all(tags) | O(N · T) | O(k) |
| ForgettingCurve::retention(t) | O(1) | O(1) |
| consolidate() | O(|STM|) | O(1) |

Where N = indexed entries, T = query tags, k = results, |STM| = entries in STM.

## Quick Start

```rust
use ternary_memory::{
    ShortTermMemory, LongTermMemory, EpisodicMemory,
    MemoryConsolidation, ForgettingCurve, ForgettingModel,
    Decision, Episode, EpisodeKind
};

// Short-term memory: 100 slots with Ebbinghaus decay
let mut stm = ShortTermMemory::with_capacity(100);

// Store decisions
stm.store(Decision::new("explore_north", 0.7, 1).with_tag("frontier"));
stm.store(Decision::new("attack_early", -0.3, 2).with_tag("combat"));
stm.store(Decision::new("trade_silk", 0.9, 3).with_tag("economy"));

// Long-term memory: accumulated statistics
let mut ltm = LongTermMemory::new();
ltm.observe("explore_north", 0.7);
ltm.observe("explore_north", 0.5);
ltm.observe("explore_north", 0.8);
// Later: retrieve statistics
if let Some(summary) = ltm.summary("explore_north") {
    println!("Mean: {:.2}, Std: {:.2}, n={}",
        summary.mean_outcome, summary.std_dev(), summary.count);
}

// Episodic memory: important events
let mut episodic = EpisodicMemory::new(1000);
episodic.store(Episode::new(
    "Found gold mine!", EpisodeKind::Breakthrough, 5, 0.95
));

// Consolidate STM → LTM + Episodic
let mut consolidation = MemoryConsolidation::new();
let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
println!("Consolidated {} entries, found {} episodes",
    result.consolidated_count, result.new_episodes);
```

## API

### Core Types

| Type | Description |
|------|-------------|
| `ShortTermMemory` | Ring buffer with Ebbinghaus decay |
| `LongTermMemory` | Welford running statistics keyed by label |
| `EpisodicMemory` | Salience-filtered event store |
| `MemoryIndex<T>` | Tag-indexed generic memory |
| `ForgettingCurve` | Configurable retention model |
| `MemoryConsolidation` | STM → LTM + Episodic transfer |

### ForgettingModel

```rust
pub enum ForgettingModel {
    Ebbinghaus { stability: f64 },
    PowerLaw { alpha: f64 },
    Linear { horizon: f64 },
}
```

## Architecture Notes

This crate implements the **η (eta) layer** cognitive substrate in the γ + η = C framework:

- **η (eta)**: Memory storage, retrieval, and consolidation algorithms. This crate provides the η-layer memory primitives that ternary agents use to learn from experience.
- **γ (gamma)**: External coordination — when to trigger consolidation, how to share memory across federated agents, distributed memory consistency. Provided by ecosystem crates (`ternary-federated`, `ternary-lease`).
- **C**: The complete agent cognitive system. γ decides when to consolidate and share; η does the actual storage and retrieval.

The ternary connection: agent decisions are evaluated as ternary outcomes (bad/neutral/good = {-1, 0, +1}), and the Ebbinghaus decay naturally weights recent ternary decisions more heavily in STM.

## References

- **Multi-Store Memory Model**: Atkinson, R.C. & Shiffrin, R.M., "Human Memory: A Proposed System and Its Control Processes," The Psychology of Learning and Motivation, 2, 89-195, 1968.
- **Episodic Memory**: Tulving, E., "Episodic and Semantic Memory," Organization of Memory, 381-403, 1972.
- **Ebbinghaus Forgetting Curve**: Ebbinghaus, H., "Über das Gedächtnis," 1885.
- **Welford's Algorithm**: Welford, B.P., "Note on a Method for Calculating Corrected Sums of Squares and Products," Technometrics, 4(3), 419-420, 1962.
- **Sleep Consolidation**: Diekelmann, S. & Born, J., "The Memory Function of Sleep," Nature Reviews Neuroscience, 11(2), 114-126, 2010.
- **Numerically Stable Variance**: Chan, T.F., Golub, G.H. & LeVeque, R.J., "Algorithms for Computing the Sample Variance," American Statistician, 37(3), 242-247, 1983.

## License

MIT
