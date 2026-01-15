"""
Unit tests for the adaptive search agent.
"""

from app.services.adaptive import AdaptiveSearchAgent, ArmStatistics, SearchStrategy


def test_arm_statistics_update() -> None:
    """Test that arm statistics update correctly."""
    stats = ArmStatistics()
    assert stats.pulls == 0
    assert stats.avg_reward == 0.0

    stats.update(100.0)
    assert stats.pulls == 1
    assert stats.avg_reward == 100.0

    stats.update(200.0)
    assert stats.pulls == 2
    assert stats.avg_reward == 150.0


def test_search_strategy_str() -> None:
    """Test SearchStrategy string representation."""
    flat = SearchStrategy(name="flat", use_index=False, nprobe=None, max_codes=None)
    assert str(flat) == "Flat"

    ivf = SearchStrategy(
        name="ivf_balanced", use_index=True, nprobe=10, max_codes=50000
    )
    assert str(ivf) == "IVF(nprobe=10, max_codes=50000)"


def test_agent_initialization(tmp_path: str) -> None:
    """Test agent initializes with correct arms and statistics."""
    agent = AdaptiveSearchAgent(
        algorithm="epsilon-greedy", epsilon=0.1, directory=str(tmp_path)
    )

    assert agent.algorithm == "epsilon-greedy"
    assert agent.epsilon == 0.1
    assert len(agent.arms) == 4
    assert len(agent.statistics) == 4
    assert agent.total_pulls == 0

    # Check all arms are initialized
    arm_names = [arm.name for arm in agent.arms]
    assert "flat" in arm_names
    assert "ivf_conservative" in arm_names
    assert "ivf_balanced" in arm_names
    assert "ivf_aggressive" in arm_names


def test_epsilon_greedy_exploration(tmp_path: str) -> None:
    """Test epsilon-greedy explores with correct probability."""
    agent = AdaptiveSearchAgent(
        algorithm="epsilon-greedy", epsilon=1.0, directory=str(tmp_path)
    )

    # With epsilon=1.0, should always explore (random selection)
    selections = [agent.select_arm().name for _ in range(100)]
    # Should have variety (not all the same)
    assert len(set(selections)) > 1


def test_epsilon_greedy_exploitation(tmp_path: str) -> None:
    """Test epsilon-greedy exploits best arm."""
    agent = AdaptiveSearchAgent(
        algorithm="epsilon-greedy", epsilon=0.0, directory=str(tmp_path)
    )

    # Update one arm to be clearly better
    agent.update("flat", latency_ms=1.0)  # High reward
    agent.update("ivf_balanced", latency_ms=100.0)  # Low reward

    # With epsilon=0.0, should always exploit (select best)
    selections = [agent.select_arm().name for _ in range(10)]
    assert all(s == "flat" for s in selections)


def test_ucb1_initialization(tmp_path: str) -> None:
    """Test UCB1 pulls each arm at least once during initialization."""
    agent = AdaptiveSearchAgent(algorithm="ucb1", epsilon=0.0, directory=str(tmp_path))

    # Pull arms until all have been tried
    selected = set()
    for _ in range(10):
        arm = agent.select_arm()
        selected.add(arm.name)
        agent.update(arm.name, latency_ms=10.0)
        if len(selected) == 4:
            break

    assert len(selected) == 4


def test_agent_update_reward_calculation(tmp_path: str) -> None:
    """Test that rewards are calculated correctly from latency."""
    agent = AdaptiveSearchAgent(directory=str(tmp_path))

    # Faster query = higher reward
    agent.update("flat", latency_ms=1.0)
    assert agent.statistics["flat"].avg_reward == 1000.0

    agent.update("ivf_balanced", latency_ms=10.0)
    assert agent.statistics["ivf_balanced"].avg_reward == 100.0

    # Flat should have higher reward (faster)
    assert (
        agent.statistics["flat"].avg_reward
        > agent.statistics["ivf_balanced"].avg_reward
    )


def test_agent_get_stats(tmp_path: str) -> None:
    """Test agent statistics reporting."""
    agent = AdaptiveSearchAgent(directory=str(tmp_path))

    # Initial stats
    stats = agent.get_stats()
    assert stats["algorithm"] == "epsilon-greedy"
    assert stats["total_pulls"] == 0
    assert "arms" in stats
    assert len(stats["arms"]) == 4

    # After some updates
    agent.update("flat", latency_ms=5.0)
    stats = agent.get_stats()
    assert stats["total_pulls"] == 1
    assert stats["arms"]["flat"]["pulls"] == 1
    assert stats["arms"]["flat"]["avg_latency_ms"] == 5.0


def test_agent_reset(tmp_path: str) -> None:
    """Test agent reset clears all statistics."""
    agent = AdaptiveSearchAgent(directory=str(tmp_path))

    # Add some data
    agent.update("flat", latency_ms=1.0)
    agent.update("ivf_balanced", latency_ms=2.0)
    assert agent.total_pulls == 2

    # Reset
    agent.reset()
    assert agent.total_pulls == 0
    for stats in agent.statistics.values():
        assert stats.pulls == 0
        assert stats.avg_reward == 0.0


def test_agent_handles_zero_latency(tmp_path: str) -> None:
    """Test agent handles edge case of zero latency."""
    agent = AdaptiveSearchAgent(directory=str(tmp_path))

    # Should not crash or update with zero latency
    agent.update("flat", latency_ms=0.0)
    assert agent.statistics["flat"].pulls == 0  # Should not update


def test_agent_state_persistence(tmp_path: str) -> None:
    """Test agent saves and loads state correctly."""
    # Create agent with temp directory
    agent1 = AdaptiveSearchAgent(directory=str(tmp_path))

    # Add some data
    agent1.update("flat", latency_ms=1.0)
    agent1.update("ivf_balanced", latency_ms=2.0)
    agent1.save_state()

    # Create new agent with same directory
    agent2 = AdaptiveSearchAgent(directory=str(tmp_path))

    # Should load previous state
    assert agent2.total_pulls == 2
    assert agent2.statistics["flat"].pulls == 1
    assert agent2.statistics["ivf_balanced"].pulls == 1
