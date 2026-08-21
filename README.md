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

**When to use this crate.** Use `ternary-memory` when you're building an agent (game NPC, robot, LLM-driven assistant, RL policy) that needs to *learn from experience* — i.e. recall recent context, accumulate statistics about what works, and remember noteworthy events. Don't use it for raw time-series storage or as a persistent database: everything lives in process memory and is lost when the process exits.

## How It Works

### Short-Term Memory: Ring Buffer with Decay

STM is a fixed-capacity ring buffer. When full, the oldest entry is overwritten. Each entry's retention is computed from the **Ebbinghaus forgetting curve**:

```
R(t) = e^(−t / S)
```

where t = elapsed time since storage, S = stability parameter (half-life analog). At t = S · ln(2) ≈ 0.693·S, retention drops to 50%. The default curve (`ForgettingCurve::default()`) is constructed via `ebbinghaus_with_half_life(100.0)`, so the default half-life is 100 ticks.

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

| n    | C(n)    |
|------|---------|
| 0    | 0.0     |
| 1    | 0.5     |
| 10   | ≈ 0.76  |
| 100  | ≈ 0.91  |
| → ∞  | → 1     |

### Episodic Memory: Salience-Filtered Events

Episodic memory stores only **noteworthy** events. The built-in consolidation pass detects two episode kinds automatically:

- **Breakthrough**: outcome ≥ `breakthrough_threshold` (default: 0.8)
- **Near-miss**: outcome ≤ `near_miss_threshold` (default: −0.5)

A third kind, **Surprise** (`EpisodeKind::Surprise`), is user-defined — `MemoryConsolidation` does not auto-detect it; construct and store those episodes yourself via `Episode::new(..., EpisodeKind::Surprise, ...)`.

When capacity is reached, the oldest episode is evicted (FIFO). This bounded design ensures episodic memory never causes unbounded memory growth. (`EpisodicMemory::new(0)` is also legal and stores nothing.)

### Memory Index: Tag-Based Retrieval

Any memory entry can be tagged with context keys for fast retrieval via the generic `MemoryIndex<T>`:

```
ContextTag = { key: String, value: String }
```

Two query modes:

- `query_all(tags)`: AND semantics — match all specified tags
- `query_any(tags)`: OR semantics — match any specified tag

Results are sorted by relevance (descending), enabling priority-weighted retrieval. Entries whose `relevance` is `NaN` are skipped.

### Memory Consolidation

Periodic consolidation transfers STM entries to LTM statistics:

```
For each decision in STM.drain():
    LTM.observe(decision.action, decision.outcome)
    If decision.outcome ≥ breakthrough_threshold:
        Episodic.store(Breakthrough episode)
    Else if decision.outcome ≤ near_miss_threshold:
        Episodic.store(NearMiss episode)
```

Breakthrough and near-miss are mutually exclusive: a single decision produces at most one episode. This implements the **sleep consolidation** hypothesis (Diekelmann & Born, 2010): short-term memories are transferred to long-term storage during quiescent periods. Use `consolidate_selective(...)` when you want to keep some decisions in STM (e.g. only consolidate decisions the agent has finished reasoning about).

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `ShortTermMemory::store(d)` | O(1) | O(1) |
| `ShortTermMemory::recall()` | O(capacity) | O(capacity) |
| `ShortTermMemory::drain()` | O(capacity) | O(capacity) |
| `LongTermMemory::observe(label, x)` | O(k) | O(1) |
| `LongTermMemory::get(label)` | O(k) | O(1) |
| `EpisodicMemory::store(e)` | O(1) amortised | O(1) |
| `EpisodicMemory::recall_top(n)` | O(m log m) | O(m) |
| `MemoryIndex::query_all(tags)` | O(N · T) | O(k) |
| `ForgettingCurve::retention(t)` | O(1) | O(1) |
| `MemoryConsolidation::consolidate()` | O(\|STM\|) | O(1) |

Where N = indexed entries, T = query tags, k = results, m = episodes stored, \|STM\| = entries drained from STM.

## Quick Start

This exact program is `examples/quickstart.rs` in the repo and is run as part of CI:

```rust
use ternary_memory::{
    Decision, Episode, EpisodeKind, EpisodicMemory, LongTermMemory,
    MemoryConsolidation, ShortTermMemory,
};

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
```

Expected output (the LTM block computes `mean(0.7, 0.5, 0.8) ≈ 0.67`, `σ ≈ 0.12`):

```
Mean: 0.67, Std: 0.12, n=3
Consolidated 3 entries, found 1 episodes
```

(`trade_silk` at outcome 0.9 is the single breakthrough; `attack_early` at −0.3 is between the two thresholds so no episode is created; `explore_north` was already moved into LTM above so it's just an observation.)

Run it yourself:

```sh
cargo run --example quickstart
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

Full per-item documentation is available via `cargo doc --open`.

## Architecture Notes

This crate implements the **η (eta) layer** cognitive substrate in the γ + η = C framework:

- **η (eta)**: Memory storage, retrieval, and consolidation algorithms. This crate provides the η-layer memory primitives that ternary agents use to learn from experience.
- **γ (gamma)**: External coordination — when to trigger consolidation, how to share memory across federated agents, distributed memory consistency. Provided by ecosystem crates (`ternary-federated`, `ternary-lease`).
- **C**: The complete agent cognitive system. γ decides when to consolidate and share; η does the actual storage and retrieval.

The ternary connection: agent decisions are evaluated as ternary outcomes (bad/neutral/good = {-1, 0, +1}), and the Ebbinghaus decay naturally weights recent ternary decisions more heavily in STM.

## Numerical-Stability & Edge-Case Guarantees

A few guarantees the crate is explicitly tested for:

- **Welford variance** matches a two-pass reference within `1e-3` even on adversarial inputs (large nearly-equal values that defeat the naive `Σx² − (Σx)²/n` formula).
- **NaN-safe ranking.** `LongTermMemory::best_label`, `EpisodicMemory::recall_top`, and `MemoryIndex::query_{all,any}` skip NaN entries instead of silently letting them win the comparison. (`partial_cmp().unwrap_or(Equal)` is never used; comparisons go through `f64::total_cmp`.)
- **Capacity zero is legal.** `ShortTermMemory::new(0, _)` and `EpisodicMemory::new(0)` are no-ops rather than panicking.
- **Complete forgetting is well-defined.** A `Linear` curve at `t ≥ horizon` returns retention exactly `0.0`; `weighted_average_outcome` then returns `0.0` rather than dividing by zero total weight.

## References

- **Multi-Store Memory Model**: Atkinson, R.C. & Shiffrin, R.M., "Human Memory: A Proposed System and Its Control Processes," The Psychology of Learning and Motivation, 2, 89-195, 1968.
- **Episodic Memory**: Tulving, E., "Episodic and Semantic Memory," Organization of Memory, 381-403, 1972.
- **Ebbinghaus Forgetting Curve**: Ebbinghaus, H., "Über das Gedächtnis," 1885.
- **Welford's Algorithm**: Welford, B.P., "Note on a Method for Calculating Corrected Sums of Squares and Products," Technometrics, 4(3), 419-420, 1962.
- **Sleep Consolidation**: Diekelmann, S. & Born, J., "The Memory Function of Sleep," Nature Reviews Neuroscience, 11(2), 114-126, 2010.
- **Numerically Stable Variance**: Chan, T.F., Golub, G.H. & LeVeque, R.J., "Algorithms for Computing the Sample Variance," American Statistician, 37(3), 242-247, 1983.

## License

MIT
