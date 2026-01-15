"""
Self-tuning search agent using Multi-Armed Bandit (MAB) algorithm.
Dynamically adapts search strategy based on observed query behavior.
"""

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import settings


@dataclass
class SearchStrategy:
    """Represents a search configuration 'arm' in the bandit."""

    name: str
    use_index: bool
    nprobe: int | None
    max_codes: int | None

    def __str__(self) -> str:
        if not self.use_index:
            return "Flat"
        return f"IVF(nprobe={self.nprobe}, max_codes={self.max_codes})"


@dataclass
class ArmStatistics:
    """Tracks performance statistics for each arm."""

    pulls: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0

    def update(self, reward: float) -> None:
        """Update statistics with new reward."""
        self.pulls += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.pulls


class AdaptiveSearchAgent:
    """
    Multi-Armed Bandit agent for adaptive search strategy selection.
    Uses epsilon-greedy or UCB1 algorithm to balance exploration and exploitation.
    """

    def __init__(
        self,
        algorithm: Literal["epsilon-greedy", "ucb1"] = "epsilon-greedy",
        epsilon: float = 0.1,
        directory: str = settings.DATA_DIR,
    ):
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.directory = directory
        self.state_path = os.path.join(directory, "agent_state.json")

        # Define search strategy arms
        self.arms: list[SearchStrategy] = [
            SearchStrategy(name="flat", use_index=False, nprobe=None, max_codes=None),
            SearchStrategy(
                name="ivf_conservative", use_index=True, nprobe=5, max_codes=10000
            ),
            SearchStrategy(
                name="ivf_balanced", use_index=True, nprobe=10, max_codes=50000
            ),
            SearchStrategy(
                name="ivf_aggressive", use_index=True, nprobe=20, max_codes=100000
            ),
        ]

        # Initialize statistics for each arm
        self.statistics: dict[str, ArmStatistics] = {
            arm.name: ArmStatistics() for arm in self.arms
        }

        self.total_pulls = 0

        # Try to load existing state
        self.load_state()

    def select_arm(self) -> SearchStrategy:
        """
        Select an arm (search strategy) using the configured algorithm.
        Returns the selected SearchStrategy.
        """
        if self.algorithm == "epsilon-greedy":
            return self._epsilon_greedy_select()
        else:
            return self._ucb1_select()

    def _epsilon_greedy_select(self) -> SearchStrategy:
        """Epsilon-greedy selection: explore with probability epsilon, else exploit."""
        if random.random() < self.epsilon:
            # Explore: choose random arm
            return random.choice(self.arms)
        else:
            # Exploit: choose arm with highest average reward
            best_arm_name = max(
                self.statistics.keys(),
                key=lambda name: self.statistics[name].avg_reward,
            )
            return next(arm for arm in self.arms if arm.name == best_arm_name)

    def _ucb1_select(self) -> SearchStrategy:
        """
        UCB1 selection: choose arm with highest upper confidence bound.
        UCB = avg_reward + sqrt(2 * ln(total_pulls) / arm_pulls)
        """
        # If any arm hasn't been pulled, pull it first (initialization)
        for arm in self.arms:
            if self.statistics[arm.name].pulls == 0:
                return arm

        # Calculate UCB for each arm
        ucb_values = {}
        for arm in self.arms:
            stats = self.statistics[arm.name]
            exploration_bonus = math.sqrt(2 * math.log(self.total_pulls) / stats.pulls)
            ucb_values[arm.name] = stats.avg_reward + exploration_bonus

        # Select arm with highest UCB
        best_arm_name = max(ucb_values.keys(), key=lambda name: ucb_values[name])
        return next(arm for arm in self.arms if arm.name == best_arm_name)

    def update(self, arm_name: str, latency_ms: float) -> None:
        """
        Update arm statistics after observing query latency.
        Reward = 1000 / latency_ms (faster queries = higher reward)
        """
        if latency_ms <= 0:
            return

        # Calculate reward (inverse latency)
        reward = 1000.0 / latency_ms

        # Update arm statistics
        self.statistics[arm_name].update(reward)
        self.total_pulls += 1

        # Periodically save state
        if self.total_pulls % 10 == 0:
            self.save_state()

    def get_stats(self) -> dict[str, Any]:
        """Return current agent statistics."""
        return {
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "total_pulls": self.total_pulls,
            "arms": {
                name: {
                    "pulls": stats.pulls,
                    "avg_reward": round(stats.avg_reward, 2),
                    "total_reward": round(stats.total_reward, 2),
                    "avg_latency_ms": (
                        round(1000.0 / stats.avg_reward, 2)
                        if stats.avg_reward > 0
                        else 0
                    ),
                }
                for name, stats in self.statistics.items()
            },
        }

    def save_state(self) -> None:
        """Persist agent state to disk."""
        state = {
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "total_pulls": self.total_pulls,
            "statistics": {
                name: {
                    "pulls": stats.pulls,
                    "total_reward": stats.total_reward,
                    "avg_reward": stats.avg_reward,
                }
                for name, stats in self.statistics.items()
            },
        }
        os.makedirs(self.directory, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> None:
        """Load agent state from disk if it exists."""
        if not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path) as f:
                state = json.load(f)

            self.algorithm = state.get("algorithm", self.algorithm)
            self.epsilon = state.get("epsilon", self.epsilon)
            self.total_pulls = state.get("total_pulls", 0)

            # Restore statistics
            for name, stats_dict in state.get("statistics", {}).items():
                if name in self.statistics:
                    self.statistics[name] = ArmStatistics(
                        pulls=stats_dict["pulls"],
                        total_reward=stats_dict["total_reward"],
                        avg_reward=stats_dict["avg_reward"],
                    )
        except Exception:
            # If loading fails, start fresh
            pass

    def reset(self) -> None:
        """Reset agent state (for re-exploration)."""
        self.statistics = {arm.name: ArmStatistics() for arm in self.arms}
        self.total_pulls = 0
        if os.path.exists(self.state_path):
            os.remove(self.state_path)


# Global agent instance
agent = AdaptiveSearchAgent(algorithm="epsilon-greedy", epsilon=0.1)
