"""Custom Solara dashboard for the CMP400 WorldModel."""

from __future__ import annotations

import contextlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TextIO

import altair as alt
import pandas as pd
import solara
from solara.server import app as solara_app
from solara.server.app import AppScript
from solara.server.starlette import ServerStarlette

import config
from src.model.world_model import WorldModel

DEFAULT_IRON = sum(config.ENV_STARTING_IRON_RANGE) / 2.0
DEFAULT_GOLD = sum(config.ENV_STARTING_GOLD_RANGE) / 2.0


class RunArtifactManager:
    """Create the same log artifacts as the CLI version for each dashboard run."""

    def __init__(self, root: str = "logs") -> None:
        self.root = Path(root)
        self.current_dir: Path | None = None
        self.chronicle_path: Path | None = None
        self.sim_log_handle: TextIO | None = None
        self.chart_data_path: Path | None = None
        self.pending_params: Dict[str, Any] | None = None

    def _allocate_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = self.root / f"webui_{timestamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = self.root / f"webui_{timestamp}_v{suffix}"
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def prepare_for_model(self, params: Dict[str, Any], previous_model: WorldModel | None) -> None:
        """Finalize any open run and stage params for the next run."""
        if self.current_dir is not None:
            self.finalize(previous_model)
        self.pending_params = dict(params)

    def ensure_run_started(self, model: WorldModel) -> None:
        if self.current_dir is not None:
            return
        params = self.pending_params or {}
        self._start_run(model, params)

    def _start_run(self, new_model: WorldModel, params: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        run_dir = self._allocate_run_dir()
        self.current_dir = run_dir
        self.chronicle_path = run_dir / "chronicle.json"
        (run_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        new_model.enable_agent_state_logging(run_dir)
        new_model.save_config_summary(str(run_dir / "config_summary.txt"))
        new_model.save_chronicle(self.chronicle_path)
        (run_dir / "starting_settings.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in params.items()),
            encoding="utf-8",
        )
        if self.sim_log_handle is not None:
            try:
                self.sim_log_handle.close()
            except Exception:
                pass
        self.sim_log_handle = (run_dir / "simulation_output.log").open("w", encoding="utf-8")
        self.chart_data_path = run_dir / "resource_history.csv"
        self.pending_params = dict(params)
        self.export_chart_data(new_model)

    def record_progress(self, model: WorldModel | None) -> None:
        if model is None or self.chronicle_path is None or self.current_dir is None:
            return
        model.save_chronicle(self.chronicle_path)

    def finalize(self, model: WorldModel | None) -> None:
        if self.current_dir is None:
            self.sim_log_handle = None
            self.chronicle_path = None
            self.chart_data_path = None
            return
        if model is not None:
            self.record_progress(model)
            self.export_chart_data(model)
            model.close_agent_state_logs()
        if self.sim_log_handle is not None:
            try:
                self.sim_log_handle.close()
            except Exception:
                pass
            self.sim_log_handle = None
        self.current_dir = None
        self.chronicle_path = None
        self.chart_data_path = None

    def append_sim_output(self, text: str) -> None:
        if not text or self.sim_log_handle is None:
            return
        self.sim_log_handle.write(text.rstrip() + "\n\n")
        self.sim_log_handle.flush()

    def export_chart_data(self, model: WorldModel | None) -> None:
        if model is None or self.chart_data_path is None or self.current_dir is None:
            return
        df = model.datacollector.get_model_vars_dataframe()
        df.to_csv(self.chart_data_path, index=True)


def default_params() -> Dict[str, Any]:
    """Return a fresh defaults dict each time."""
    return {
        "initial_wealth": float(config.STARTING_WEALTH),
        "initial_food": float(config.STARTING_FOOD),
        "initial_wood": float(config.STARTING_WOOD),
        "initial_iron": float(DEFAULT_IRON),
        "initial_gold": float(DEFAULT_GOLD),
        "population_E": int(config.STARTING_POPULATION),
        "population_W": int(config.STARTING_POPULATION),
        "use_llm": True,
        "seed": 42,
    }


@dataclass(frozen=True)
class SliderSpec:
    """Describe a numeric slider."""

    name: str
    label: str
    min_value: float
    max_value: float
    step: float
    is_int: bool = False

    def format_value(self, value: float | int) -> str:
        return f"{int(value)}" if self.is_int else f"{value:.2f}"

    def coerce(self, value: float) -> float | int:
        return int(round(value)) if self.is_int else float(value)


SLIDER_SPECS: List[SliderSpec] = [
    SliderSpec("initial_wealth", "Initial Wealth", 0.0, 20.0, 0.5),
    SliderSpec("initial_food", "Initial Food", 0.0, 20.0, 0.5),
    SliderSpec("initial_wood", "Initial Wood", 0.0, 20.0, 0.5),
    SliderSpec("initial_iron", "Initial Iron", 0.0, 10.0, 0.5),
    SliderSpec("initial_gold", "Initial Gold", 0.0, 10.0, 0.5),
    SliderSpec("population_E", "East Population", 10, 500, 10, True),
    SliderSpec("population_W", "West Population", 10, 500, 10, True),
]

SERIES = [
    "wealth_E",
    "wealth_W",
    "food_E",
    "food_W",
    "wood_E",
    "wood_W",
    "iron_E",
    "iron_W",
    "gold_E",
    "gold_W",
]


def build_model(params: Dict[str, Any]) -> WorldModel:
    """Instantiate the WorldModel with the chosen knobs."""
    seed_value = params.get("seed")
    try:
        random_seed = int(seed_value)
    except (TypeError, ValueError):
        random_seed = None
    return WorldModel(
        random_seed=random_seed,
        initial_wealth=params["initial_wealth"],
        initial_food=params["initial_food"],
        initial_wood=params["initial_wood"],
        initial_iron=params["initial_iron"],
        initial_gold=params["initial_gold"],
        population_E=params["population_E"],
        population_W=params["population_W"],
        use_llm=bool(params["use_llm"]),
    )


def capture_step_output(model: WorldModel) -> str:
    """Run one step and return whatever stdout the model produced."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.step()
    return buffer.getvalue().strip()


@solara.component
def ParameterControls(params: Dict[str, Any], on_change) -> None:
    """Render all numeric sliders plus the LLM toggle."""
    with solara.Card("Starting Conditions", margin=0):
        for spec in SLIDER_SPECS:
            value = params[spec.name]
            with solara.Column(gap="0.25rem"):
                solara.Text(f"{spec.label}: {spec.format_value(value)}")
                slider_cls = solara.SliderInt if spec.is_int else solara.SliderFloat

                def handle(new_value, spec_name=spec.name, convert=spec.coerce):
                    on_change(spec_name, convert(new_value))

                slider_cls(
                    label="",
                    value=value,
                    min=spec.min_value,
                    max=spec.max_value,
                    step=spec.step,
                    on_value=handle,
                )
        solara.Switch(
            label="Enable LLM Decisions",
            value=params["use_llm"],
            on_value=lambda v: on_change("use_llm", bool(v)),
        )
        solara.Text(
            f"LLM status: {'ON' if params['use_llm'] else 'OFF'}",
            style="font-weight: bold;",
        )
        def _update_seed(value: int) -> None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = 0
            on_change("seed", parsed)
        solara.InputInt(
            label="Random Seed",
            value=params.get("seed", 0),
            on_value=_update_seed,
        )


@solara.component
def ResourceCharts(model: WorldModel | None, refresh_token: int, orientation: str = "horizontal") -> None:  # noqa: ARG001
    """Render per-territory charts for wealth/food trends."""
    if model is None:
        solara.Text("Chart unavailable until the model is initialised.")
        return

    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        solara.Text("Collect a few steps to populate the chart.")
        return

    reset_df = df.reset_index().rename(columns={"index": "Step"})
    available_vars = [col for col in SERIES if col in reset_df.columns]
    tidy = (
        reset_df.melt(id_vars="Step", value_vars=available_vars, var_name="Metric", value_name="Value")
        if available_vars
        else reset_df.assign(Metric=[], Value=[])
    )
    parts = tidy["Metric"].str.split("_", expand=True)
    if parts.shape[1] >= 2:
        tidy["Series"] = parts[0].str.capitalize()
        tidy["Territory"] = parts[1].str.upper()
    else:
        tidy["Series"] = tidy["Metric"].str.capitalize()
        tidy["Territory"] = "E"

    def _render_chart(code: str, title: str):
        subset = tidy[tidy["Territory"].str.upper() == code.upper()]
        if subset.empty:
            return solara.Text(f"No data for {title}.")
        chart = (
            alt.Chart(subset)
            .mark_line()
            .encode(
                x=alt.X("Step:Q", title="Step"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color("Series:N", title="Series"),
            )
            .properties(width=320, height=320)
        )
        with solara.Card(f"{title} Resource Trends", margin=0):
            solara.FigureAltair(chart)

    if orientation == "vertical":
        with solara.Column(gap="1rem"):
            _render_chart("E", "East")
            _render_chart("W", "West")
    else:
        with solara.ColumnsResponsive(12, large=6):
            _render_chart("E", "East")
            _render_chart("W", "West")


@solara.component
def TerritorySnapshot(model: WorldModel | None) -> None:
    """Display quick resource stats for both territories."""
    if model is None:
        solara.Text("No model active.")
        return

    def format_territory(name: str, territory) -> str:
        return (
            f"**{name}**  \n"
            f"Food: {territory.food:.2f}  |  Wealth: {territory.wealth:.2f}  \n"
            f"Population: {territory.population:.1f}  |  Infrastructure: {territory.infrastructure_level}"
        )

    solara.Markdown(format_territory("East", model.east))
    solara.Markdown(format_territory("West", model.west))


@solara.component
def LogPanel(logs: List[str]) -> None:
    """Show captured stdout chunks."""
    with solara.Card(
        "Simulation Output",
        margin=0,
        style={"height": "100%", "overflow": "auto"},
    ):
        if not logs:
            solara.Text("No output yet. Run a step to see the simulation log.")
            return
        for idx, entry in enumerate(logs, start=1):
            solara.Markdown(f"#### Log {idx}\n```\n{entry}\n```")


@solara.component
def Dashboard() -> None:
    """Primary Solara component."""
    params, set_params = solara.use_state(default_params())
    logs, set_logs = solara.use_state([])  # list[str]
    steps_to_run, set_steps_to_run = solara.use_state(1)
    refresh_token, set_refresh_token = solara.use_state(0)
    steps_queue, set_steps_queue = solara.use_state(0)
    is_running, set_is_running = solara.use_state(False)
    stop_requested, set_stop_requested = solara.use_state(False)
    artifact_manager = solara.use_memo(lambda: RunArtifactManager(), [])
    model_ref = solara.use_ref(None)
    if model_ref.current is None:
        initial_model = build_model(params)
        artifact_manager.prepare_for_model(params, None)
        model_ref.current = initial_model

    solara.use_effect(lambda: (lambda: artifact_manager.finalize(model_ref.current)), [])

    def replace_model(new_params: Dict[str, Any]) -> None:
        artifact_manager.prepare_for_model(new_params, model_ref.current)
        new_model = build_model(new_params)
        model_ref.current = new_model
        set_steps_queue(0)
        set_is_running(False)
        set_stop_requested(False)

    def trigger_refresh() -> None:
        set_refresh_token(lambda value: value + 1)

    def update_params(name: str, value: Any) -> None:
        new_params = {**params, name: value}
        set_params(new_params)
        replace_model(new_params)
        set_logs([])
        trigger_refresh()

    def reset_model() -> None:
        replace_model(params)
        set_logs([])
        trigger_refresh()

    def append_log_entry(entry: str) -> None:
        if not entry:
            return

        def _update(prev: List[str]) -> List[str]:
            updated = prev + [entry]
            return updated[-200:]

        set_logs(_update)

    def run_single_step() -> None:
        if model_ref.current is None:
            return
        artifact_manager.ensure_run_started(model_ref.current)
        output = capture_step_output(model_ref.current)
        if output:
            append_log_entry(output)
            artifact_manager.append_sim_output(output)
        artifact_manager.record_progress(model_ref.current)
        artifact_manager.export_chart_data(model_ref.current)
        trigger_refresh()

    def queue_steps(count: int) -> None:
        if model_ref.current is None or count <= 0 or is_running:
            return
        set_stop_requested(False)
        set_steps_queue(count)
        set_is_running(True)

    def execute_steps() -> None:
        queue_steps(steps_to_run)

    def stop_simulation() -> None:
        set_stop_requested(True)

    def close_program() -> None:
        artifact_manager.finalize(model_ref.current)
        os._exit(0)

    def process_queue() -> None:
        if model_ref.current is None:
            return
        if steps_queue <= 0:
            if is_running:
                set_is_running(False)
            return
        if stop_requested:
            artifact_manager.record_progress(model_ref.current)
            set_steps_queue(0)
            return
        run_single_step()
        if model_ref.current.all_territories_dead():
            set_steps_queue(0)
            return
        set_steps_queue(steps_queue - 1)

    solara.use_effect(process_queue, [steps_queue, stop_requested])

    with solara.Column(gap="1rem", style={"padding": "0 1rem"}):
        solara.Markdown("## CMP400 WorldModel Dashboard")
        current_step = model_ref.current.steps if model_ref.current else 0
        solara.Text(f"Current Step: {current_step}")

        content_style = {
            "display": "flex",
            "flex-direction": "row",
            "align-items": "flex-start",
            "gap": "1rem",
            "flex-wrap": "nowrap",
        }

        with solara.Row(style=content_style):
            with solara.Column(gap="1rem", style={"flex": "0.65 1 0", "min-width": "300px"}):
                ParameterControls(params, update_params)
                with solara.Card("Simulation Controls", margin=0):
                    def update_steps(value: int) -> None:
                        try:
                            parsed = int(value)
                        except (TypeError, ValueError):
                            parsed = 1
                        parsed = max(1, min(50, parsed))
                        set_steps_to_run(parsed)

                    solara.InputInt(
                        label="Steps per run (1-50)",
                        value=steps_to_run,
                        on_value=update_steps,
                        disabled=is_running,
                    )
                    solara.Button("Run Steps", on_click=execute_steps, color="primary", disabled=is_running)
                    solara.Button("Step Once", on_click=lambda: queue_steps(1), disabled=is_running)
                    solara.Button("Stop", on_click=stop_simulation, color="warning", disabled=not is_running)
                    solara.Button("Reset Model", on_click=reset_model, color="secondary", disabled=is_running)
                    solara.Button("Close Server", on_click=close_program, color="danger")
                    if model_ref.current and model_ref.current.all_territories_dead():
                        solara.Text("All territories have collapsed. Reset to start over.", style="color: #b22222;")
                    status_text = "Running" if is_running and steps_queue > 0 else "Idle"
                    solara.Text(f"Status: {status_text}")
                    pending_text = f"{steps_queue} step(s) remaining" if steps_queue > 0 else "No pending steps"
                    solara.Text(pending_text)
                    if artifact_manager.current_dir:
                        solara.Text(
                            f"Logging to {artifact_manager.current_dir}",
                            style="font-size: 0.85em; color: #555;",
                        )
                TerritorySnapshot(model_ref.current)
                with solara.Card("Run Notes", margin=0):
                    solara.Text(
                        "Simulation output updates every step. Use Stop to cancel a run or Close Server to exit the dashboard."
                    )
            with solara.Column(gap="1rem", style={"flex": "0.35 1 0", "min-width": "260px"}):
                solara.Markdown("### Resource Trends")
                ResourceCharts(model_ref.current, refresh_token, orientation="vertical")
            with solara.Column(
                gap="0",
                style={
                    "flex": "1 1 0",
                    "min-width": "320px",
                    "max-height": "calc(100vh - 80px)",
                    "display": "flex",
                },
            ):
                LogPanel(logs)


Page = Dashboard


def ensure_solara_app_registered() -> None:
    """Register the Page component with Solara's server loader."""
    if "__default__" not in solara_app.apps:
        solara_app.apps["__default__"] = AppScript("run:Page")


def launch(port: int = 8521, host: str = "127.0.0.1", open_browser: bool = True) -> None:
    """Start the Solara server on the requested host/port."""
    ensure_solara_app_registered()
    server = ServerStarlette(port=port, host=host)
    url = f"http://{host}:{port}"
    print(f"Solara UI available at {url}")
    if open_browser:
        with contextlib.suppress(Exception):
            import webbrowser

            webbrowser.open(url)
    server.serve()


if __name__ == "__main__":
    launch()
