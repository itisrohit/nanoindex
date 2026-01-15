import json
import subprocess
import threading
from pathlib import Path
from typing import cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Label, Markdown, Static


class SectionTitle(Static):
    def __init__(self, title: str):
        super().__init__(title)
        self.styles.content_align = ("center", "middle")
        self.styles.background = "dodgerblue"
        self.styles.color = "white"
        self.styles.height = 3
        self.styles.margin = (0, 0, 1, 0)
        self.styles.text_style = "bold"


class TestRunnerPanel(Static):
    def compose(self) -> ComposeResult:
        yield SectionTitle("🧪 Unit Tests Status")
        yield Label("Running tests...", id="test-status")
        yield DataTable(id="test-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Test Case", "Status", "Outcome")
        # Run tests in background
        threading.Thread(target=self.run_tests, daemon=True).start()

    def run_tests(self) -> None:
        # Run pytest with -v to get list
        try:
            cmd = ["uv", "run", "pytest", "-v"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout

            # Simple parsing of verbose pytest output
            lines = output.splitlines()

            cleaned_results = []
            for line in lines:
                if "::" in line:
                    parts = line.split("::")
                    if len(parts) > 1:
                        test_part = parts[1]
                        # "test_name PASSED [ 50%]"
                        name_status = test_part.split()
                        if len(name_status) >= 2:
                            name = name_status[0]
                            status = name_status[1]
                            outcome = "✅ Pass" if "PASSED" in status else "❌ Fail"
                            cleaned_results.append((name, status, outcome))

            # Update UI on main thread
            self.app.call_from_thread(
                self.update_table, cleaned_results, result.returncode
            )

        except Exception as e:
            self.app.call_from_thread(self.update_error, str(e))

    def update_table(
        self, results: list[tuple[str, str, str]], return_code: int
    ) -> None:
        status_label = cast(Label, self.query_one("#test-status"))
        if return_code == 0:
            status_label.update("All Systems Operational")
            status_label.styles.color = "green"
        else:
            status_label.update("Tests Failed")
            status_label.styles.color = "red"

        table = self.query_one(DataTable)
        table.clear()
        for name, status, outcome in results:
            styled_outcome = Text(outcome)
            styled_outcome.stylize("green" if "Pass" in outcome else "red")
            table.add_row(name, status, styled_outcome)

    def update_error(self, msg: str) -> None:
        cast(Label, self.query_one("#test-status")).update(f"Error: {msg}")


class AgentStatsPanel(Static):
    def compose(self) -> ComposeResult:
        yield SectionTitle("🤖 Agent Telemetry")
        yield Label("Loading agent state...", id="agent-status")
        yield DataTable(id="agent-table")
        yield Label("", id="agent-summary")

    def on_mount(self) -> None:
        self.load_stats()
        # Auto-refresh every 5 seconds
        self.set_interval(5.0, self.load_stats)

    def load_stats(self) -> None:
        path = Path("data/agent_state.json")
        if not path.exists():
            cast(Label, self.query_one("#agent-status")).update(
                "No agent state found (Run queries first)"
            )
            return

        try:
            with open(path) as f:
                state = json.load(f)

            cast(Label, self.query_one("#agent-status")).update(
                f"Session Id: {id(state)}"
            )

            # Summary
            total_pulls = state.get("total_pulls", 0)
            epsilon = state.get("epsilon", 0.0)
            summary = f"Total Pulls: {total_pulls} | Exploration Rate (ε): {epsilon}"
            cast(Label, self.query_one("#agent-summary")).update(summary)

            # Arms Table
            table = self.query_one(DataTable)
            if len(table.columns) == 0:
                table.add_columns("Strategy", "Pulls", "Avg Reward", "Est. Latency")

            table.clear()
            arms = state.get("arms", {})
            for name, stats in arms.items():
                avg_reward = stats.get("avg_reward", 0.0)
                # Est latency = 1000 / reward (if reward > 0)
                est_latency = (
                    f"{1000 / avg_reward:.2f}ms" if avg_reward > 0.001 else "N/A"
                )

                table.add_row(
                    name, str(stats.get("pulls", 0)), f"{avg_reward:.2f}", est_latency
                )

        except Exception:
            pass


class BenchmarkPanel(Static):
    def compose(self) -> ComposeResult:
        yield SectionTitle("📊 Performance Benchmarks")
        yield Markdown(self.get_benchmark_md())

    def get_benchmark_md(self) -> str:
        # Read from benchmarks.md specifically the Ablation Study table
        path = Path("docs/benchmarks.md")
        if not path.exists():
            return "Benchmarks file not found."

        try:
            content = path.read_text()
            # Extract Ablation Study table
            # Looking for: | Configuration | ...
            start = content.find("| Configuration |")
            if start == -1:
                return "Table not found in docs/benchmarks.md"

            # Find end of table (double newline or next header)
            end = content.find("\n\n*Conclusion", start)
            if end == -1:
                end = content.find("\n\n", start + 200)  # Fallback

            table_md = content[start:end]
            return f"**Ablation Study (150k Vectors)**\n\n{table_md}"
        except Exception as e:
            return f"Error loading benchmarks: {e}"


class NanoDashboard(App[None]):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    TestRunnerPanel {
        row-span: 2;
        background: $surface-darken-1;
        border: solid $accent;
    }
    
    AgentStatsPanel {
        background: $surface-darken-1;
        border: solid green;
    }
    
    BenchmarkPanel {
        background: $surface-darken-1;
        border: solid orange;
    }
    
    Label {
        padding: 1;
    }
    
    DataTable {
        height: 1fr;
    }
    """

    TITLE = "NanoIndex Dashboard"
    SUB_TITLE = "Visual Learning & Diagnostics Console"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TestRunnerPanel()
        yield AgentStatsPanel()
        yield BenchmarkPanel()
        yield Footer()


if __name__ == "__main__":
    app = NanoDashboard()
    app.run()
