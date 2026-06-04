# Future Integration: ternary-memory

## Current State
Provides multi-layer memory systems for ternary agents: `ShortTermMemory` for recent decisions, `LongTermMemory` for persistent labels and weights, `EpisodicMemory` for significant events (breakthroughs, near-misses), `MemoryConsolidation` that periodically consolidates short-term into long-term and episodic, and `MemoryIndex` with context tags for fast retrieval.

## Integration Opportunities

### With ternary-cell (Cross-Room Persistence)
ternary-cell's tick cycle is stateless between sessions — cells wake up fresh each time. ternary-memory provides persistence: short-term memory stores the last N ticks' states, long-term memory stores learned cell behaviors (which states lead to which outcomes), and episodic memory records significant events (phase transitions, sudden surprise spikes). When a room (Codespace) restarts, cells reload their memory and resume rather than relearn from scratch.

### With ternary-transfer (Memory Transfer)
ternary-transfer moves knowledge between tasks. ternary-memory provides the knowledge representation. `LongTermMemory::weights()` maps to `KnowledgeMatrix` in ternary-transfer. Transferring knowledge between rooms means copying (or blending) long-term memory weights. `MemoryConsolidation` ensures only stable, well-consolidated knowledge is transferred — preventing ephemeral noise from propagating.

### With ternary-replay (Memory-Guided Replay)
ternary-replay reconstructs experiment histories. ternary-memory provides the context: `EpisodicMemory` tags significant replays as episodes (breakthrough moments, failure modes). When replaying, the engine uses `MemoryIndex` to find similar historical situations via context tags, enabling "last time we saw this pattern, the outcome was X" reasoning.

## Potential in Mature Systems
In room-as-codespace, each room has a ternary-memory instance that persists across Codespace restarts (stored in PLATO's tile store). When an agent enters a room, the room's memory is loaded into the Codespace. Short-term memory provides recent context. Long-term memory provides learned room behavior. Episodic memory provides landmark events. When the agent leaves, `MemoryConsolidation` runs: recent short-term memories are evaluated for promotion to long-term or episodic storage.

## Cross-Pollination Ideas
- **ternary-curriculum**: Curriculum learning sequences memory — early lessons build short-term memories that consolidate into long-term foundations for later lessons.
- **ternary-kalman**: Kalman filter as memory — the state estimate IS the long-term memory, the Kalman gain controls consolidation rate (high gain = fast learning, low gain = stable memory).
- **ternary-fuzzy**: Fuzzy memory retrieval — use fuzzy membership functions to retrieve memories that approximately match a query, not just exact context tag matches.

## Dependencies for Next Steps
- Define `RoomMemory` wrapping ternary-memory with room-specific context tags
- Add memory serialization to ternary-protocol for cross-room transfer
- Implement `MemoryConsolidation` trigger in ternary-cell's GC phase
- Benchmark memory retrieval on ESP32 with typical tag counts (10-100)
