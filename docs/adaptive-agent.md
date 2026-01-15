# Self-Tuning Search Agent

## Overview

NanoIndex includes a **self-tuning search agent** that dynamically adapts search strategy based on observed query behavior and latency constraints. This is an **agentic system** (not LLM-based) that uses **Multi-Armed Bandit (MAB)** algorithms to automatically select the optimal search configuration for each query.

## Core Concept

The agent treats different search configurations as "arms" in a multi-armed bandit problem:
- Each arm represents a specific search strategy (e.g., Flat, IVF with different parameters)
- The agent learns which arms perform best by observing query latency
- It balances **exploration** (trying new strategies) with **exploitation** (using known good strategies)

## Search Strategy Arms

The agent manages four distinct search strategies:

| Arm | Description | Use Case |
|-----|-------------|----------|
| **Flat** | Exhaustive search, no index | Small datasets, high accuracy requirements |
| **IVF Conservative** | `nprobe=5`, `max_codes=10k` | Fast queries, acceptable recall trade-off |
| **IVF Balanced** | `nprobe=10`, `max_codes=50k` | Default balanced approach |
| **IVF Aggressive** | `nprobe=20`, `max_codes=100k` | Higher recall, willing to trade latency |

## Algorithms

### Epsilon-Greedy (Default)

The epsilon-greedy algorithm balances exploration and exploitation with a simple rule:
- **Exploration (ε = 10%)**: Select a random arm
- **Exploitation (90%)**: Select the arm with highest average reward

**Reward Calculation:**
```
reward = 1000 / latency_ms
```
Faster queries receive higher rewards, incentivizing the agent to prefer low-latency strategies.

### UCB1 (Upper Confidence Bound)

UCB1 uses a more sophisticated approach based on confidence intervals:
```
UCB(arm) = avg_reward + sqrt(2 * ln(total_pulls) / arm_pulls)
```

This formula naturally balances:
- **Exploitation**: `avg_reward` term favors well-performing arms
- **Exploration**: Confidence bonus increases for less-tried arms

## Usage

### Enable Adaptive Agent

```bash
POST /api/v1/search
{
  "vector": [0.1, 0.2, 0.3],
  "top_k": 10,
  "use_agent": true
}
```

**Response:**
```json
{
  "query_id": "default",
  "results": [...],
  "latency_ms": 1.23,
  "strategy": "ivf_balanced"
}
```

The `strategy` field indicates which arm the agent selected.

### Monitor Agent Statistics

```bash
GET /api/v1/agent/stats
```

**Response:**
```json
{
  "algorithm": "epsilon-greedy",
  "epsilon": 0.1,
  "total_pulls": 100,
  "arms": {
    "flat": {
      "pulls": 45,
      "avg_reward": 9144.79,
      "total_reward": 411515.55,
      "avg_latency_ms": 0.11
    },
    "ivf_conservative": {
      "pulls": 15,
      "avg_reward": 12500.0,
      "total_reward": 187500.0,
      "avg_latency_ms": 0.08
    },
    ...
  }
}
```

### Reset Agent State

To force re-exploration (useful after data distribution changes):

```bash
POST /api/v1/agent/reset
```

## How It Works

### 1. Initialization
- Agent starts with zero knowledge about arm performance
- All arms have equal probability of selection initially

### 2. Query Processing
```python
# Agent selects strategy
strategy = agent.select_arm()

# Execute search with selected strategy
results, latency = search(query, strategy)

# Update agent with observed latency
agent.update(strategy.name, latency)
```

### 3. Learning Loop
- **Epsilon-Greedy**: 90% of the time, selects best-performing arm; 10% explores randomly
- **UCB1**: Always selects arm with highest upper confidence bound
- Statistics update after every query
- State persists to disk every 10 queries

### 4. Adaptation
Over time, the agent learns:
- Which strategies work best for the current workload
- When to use flat vs IVF search
- Optimal IVF parameters (nprobe, max_codes)

## Example Scenario

**Initial State** (Cold Start):
```
Flat: 0 pulls, 0 avg_reward
IVF Conservative: 0 pulls, 0 avg_reward
IVF Balanced: 0 pulls, 0 avg_reward
IVF Aggressive: 0 pulls, 0 avg_reward
```

**After 100 Queries**:
```
Flat: 40 pulls, 0.15ms avg latency
IVF Conservative: 20 pulls, 0.08ms avg latency  ← Best performer
IVF Balanced: 30 pulls, 0.12ms avg latency
IVF Aggressive: 10 pulls, 0.20ms avg latency
```

**Agent Behavior**: Now preferentially selects IVF Conservative (90% of the time) while still exploring other options (10%).

## Benefits

1. **Zero Configuration**: No manual tuning of search parameters required
2. **Workload Adaptive**: Automatically adjusts to query patterns and data distribution
3. **Continuous Learning**: Performance improves over time as more queries are processed
4. **Graceful Degradation**: Falls back to exploration if performance degrades

## Implementation Details

### State Persistence

Agent state is saved to `data/agent_state.json`:
```json
{
  "algorithm": "epsilon-greedy",
  "epsilon": 0.1,
  "total_pulls": 100,
  "statistics": {
    "flat": {
      "pulls": 45,
      "total_reward": 411515.55,
      "avg_reward": 9144.79
    },
    ...
  }
}
```

This ensures the agent retains learned knowledge across server restarts.

### Performance Overhead

The agent adds minimal overhead:
- **Selection**: O(k) where k = number of arms (4)
- **Update**: O(1) per query
- **Typical overhead**: < 0.01ms per query

## Configuration

The agent can be configured at initialization:

```python
from app.services.adaptive import AdaptiveSearchAgent

# Epsilon-greedy with custom exploration rate
agent = AdaptiveSearchAgent(
    algorithm="epsilon-greedy",
    epsilon=0.2  # 20% exploration
)

# UCB1 algorithm
agent = AdaptiveSearchAgent(
    algorithm="ucb1"
)
```

## When to Use

**Use the adaptive agent when:**
- Query patterns are unknown or variable
- You want automatic optimization without manual tuning
- Workload characteristics change over time
- You need to balance latency and recall dynamically

**Use manual strategy selection when:**
- You have strict latency SLAs that require predictable behavior
- Query patterns are well-understood and static
- You need deterministic search behavior for testing

## Future Enhancements

Potential improvements to the agent:
- **Contextual Bandits**: Consider query characteristics (vector norm, top-k value)
- **Thompson Sampling**: Bayesian approach for better exploration
- **Latency SLA Awareness**: Reject strategies that violate latency constraints
- **Recall Tracking**: Incorporate quality metrics alongside latency
- **Adaptive Epsilon**: Decay exploration rate over time

---

**Key Insight**: The self-tuning agent transforms NanoIndex from a static search engine into an **adaptive system** that continuously optimizes itself based on real-world usage patterns.
