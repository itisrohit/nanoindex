"""
Integration tests for adaptive agent convergence.
Simulates query patterns to verify the agent learns optimal strategies.
"""

from app.services.adaptive import AdaptiveSearchAgent


def test_agent_convergence_to_fastest_arm(tmp_path: str) -> None:
    """
    Test that the agent converges to the arm with the lowest latency (highest reward).
    """
    agent = AdaptiveSearchAgent(
        algorithm="epsilon-greedy", epsilon=0.1, directory=str(tmp_path)
    )

    # Define synthetic latencies for each arm
    # "ivf_conservative" is the winner here (10ms)
    latencies = {
        "flat": 50.0,
        "ivf_conservative": 10.0,
        "ivf_balanced": 20.0,
        "ivf_aggressive": 100.0,
    }

    # Simulate 1000 queries
    for _ in range(1000):
        # 1. Select arm
        strategy = agent.select_arm()

        # 2. "Execute" query (lookup synthetic latency)
        latency = latencies.get(strategy.name, 50.0)

        # 3. Update agent
        agent.update(strategy.name, latency)

    # Check statistics
    stats = agent.get_stats()
    arms = stats["arms"]

    # Verify "ivf_conservative" is the most pulled arm
    most_pulled_arm = max(arms.items(), key=lambda x: x[1]["pulls"])[0]

    assert most_pulled_arm == "ivf_conservative", (
        f"Agent failed to converge. Stats: {arms}"
    )

    # Verify it was pulled significantly more than others (exploitation)
    total_pulls = agent.total_pulls
    conservative_pulls = arms["ivf_conservative"]["pulls"]
    exploitation_rate = conservative_pulls / total_pulls

    # With epsilon=0.1, we expect ~90% exploitation + random share of exploration
    # Being safe: at least 70% should be the best arm
    assert exploitation_rate > 0.70, (
        f"Exploitation rate too low: {exploitation_rate:.2f}"
    )


def test_agent_ucb1_exploration(tmp_path: str) -> None:
    """
    Test that UCB1 explicitly explores under-pulled arms.
    """
    agent = AdaptiveSearchAgent(algorithm="ucb1", directory=str(tmp_path))

    # All arms start equal.
    # We pull "flat" 10 times with mediocre latency.
    for _ in range(10):
        agent.update("flat", latency_ms=50.0)

    # Now the other arms have 0 pulls. UCB1 MUST select them to reduce uncertainty.
    # The bonus +Infinity for 0 pulls ensures they are picked next.

    next_arms = set()
    for _ in range(10):
        strategy = agent.select_arm()
        next_arms.add(strategy.name)
        # Update with some latency so it doesn't stay at 0 pulls
        agent.update(strategy.name, latency_ms=50.0)

    # It should have tried the other arms
    assert "ivf_conservative" in next_arms
    assert "ivf_balanced" in next_arms
    assert "ivf_aggressive" in next_arms
