"""Population: N agents with distinct genomes/seeds, evaluated identically.

Fairness rules encoded here:
- every agent gets the same data slices, costs, risk baseline, timesteps and
  scoring window - the ONLY differences are its genome and its training seed
- every agent's genome and result row is persisted before/after evaluation,
  so a crashed run still leaves a complete roster behind
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from darwin.environment.simulator import SimulatorConfig
from darwin.evolution.fitness import FitnessConfig, compute_fitness
from darwin.evolution.genome import Genome
from darwin.evolution.survival import SurvivalConfig, Verdict, evaluate_survival
from darwin.execution.risk import RiskConfig
from darwin.experiments.tracker import (
    record_agent,
    record_genome,
    update_agent_result,
)
from darwin.experiments.training import train_and_evaluate


@dataclass(frozen=True)
class AgentSpec:
    """One roster entry: identity + genome + training seed."""

    agent_id: str
    genome: Genome
    seed: int
    generation: int = 0


_AGGREGABLE = (
    "total_return", "max_drawdown", "sharpe", "sortino", "profit_factor",
    "win_rate", "n_trades", "fees_paid", "avg_trade_net", "exposure",
    "fitness",
)


def aggregate_seed_metrics(seed_runs: list[dict]) -> dict:
    """Mean across seed runs for numeric metrics; full per-seed detail kept."""
    if not seed_runs:
        msg = "cannot aggregate zero seed runs"
        raise ValueError(msg)
    out: dict = {"per_seed": seed_runs}
    for key in _AGGREGABLE:
        values = [run[key] for run in seed_runs if run.get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    return out


class Population:
    """Creates and tracks a generation-0 roster of agents."""

    def __init__(self, size: int, db_path: Any = "experiments/metadata.sqlite") -> None:
        if size < 2:
            msg = "population size must be >= 2 (selection needs rivals)"
            raise ValueError(msg)
        self.size = size
        self.db_path = db_path
        self.agents: list[AgentSpec] = []

    def initialize(self, master_seed: int) -> list[AgentSpec]:
        """Create the roster deterministically from one master seed."""
        rng = np.random.default_rng(master_seed)
        self.agents = []
        for i in range(self.size):
            genome_id = uuid.uuid4().hex[:10]
            genome = Genome.random(rng, genome_id=genome_id)
            record_genome(
                genome.values,
                genome_id=genome_id,
                parent_id=None,
                generation=0,
                mutations=[],
                db_path=self.db_path,
            )
            agent_id = f"a{master_seed:05d}-{i:03d}"
            seed = int(rng.integers(0, 2**31 - 1))
            record_agent(
                agent_id,
                genome_id=genome_id,
                seed=seed,
                generation=0,
                db_path=self.db_path,
            )
            self.agents.append(
                AgentSpec(agent_id=agent_id, genome=genome, seed=seed)
            )
        return self.agents

    def record_result(
        self,
        agent_id: str,
        metrics: dict,
        model_path: str | None = None,
        status: str = "evaluated",
    ) -> None:
        """Persist metrics/status; model_path=None preserves the stored one."""
        update_agent_result(
            agent_id, metrics=metrics, model_path=model_path,
            status=status, db_path=self.db_path,
        )

    def evaluate_agent(
        self,
        agent: AgentSpec,
        *,
        ohlcv: Any,
        feats: Any,
        timeframe: str,
        sim_cfg: SimulatorConfig,
        risk_cfg: RiskConfig,
        fitness_cfg: FitnessConfig,
        survival_cfg: SurvivalConfig,
        train_end: int,
        val_end: int,
        timesteps: int,
        eval_window: int = 3_000,
        score_window: int | None = None,
        out_dir: str = "experiments/runs",
        inherit_weights_from: str | None = None,
        n_seeds: int = 1,
        policy: str = "MlpPolicy",
        reward_baseline_weight: float = 0.0,
    ) -> tuple[Any, dict, Verdict]:
        """Train, score, and record one agent. Returns (report, metrics, verdict).

        The single funnel every agent - founder or child - passes through, so
        selection never compares apples to oranges. Behavior fingerprint is
        folded into the persisted metrics (zero extra compute). With
        ``n_seeds > 1`` the agent trains under several seeds and metrics are
        the MEAN across seeds - single-seed luck stops selecting.
        """
        seed_runs: list[dict] = []
        behavior: dict = {}
        report = None
        for k in range(max(1, n_seeds)):
            model_path, seed_report, behavior = train_and_evaluate(
                seed=agent.seed + k,
                ohlcv=ohlcv,
                feats=feats,
                timeframe=timeframe,
                sim_cfg=sim_cfg,
                risk_cfg=risk_cfg,
                train_end=train_end,
                val_end=val_end,
                timesteps=timesteps,
                eval_window=eval_window,
                out_dir=out_dir,
                genome=agent.genome,
                score_window_bars=score_window,
                init_from_model_path=inherit_weights_from,
                reward_baseline_weight=reward_baseline_weight,
                policy=policy,
            )
            m = vars(seed_report)
            m["fitness"] = compute_fitness(
                {**m, "initial_capital_proxy": sim_cfg.initial_capital},
                fitness_cfg,
            ).total
            seed_runs.append(m)
            report = seed_report

        metrics = aggregate_seed_metrics(seed_runs)
        metrics["behavior"] = behavior
        verdict = evaluate_survival(metrics, metrics["fitness"], survival_cfg)
        self.record_result(agent.agent_id, metrics=metrics, model_path=model_path)
        from darwin.experiments.tracker import mark_agent_status

        mark_agent_status(
            agent.agent_id,
            verdict.status,
            reason="; ".join(verdict.reasons) if verdict.reasons else None,
            fitness=metrics["fitness"],
            max_drawdown=report.max_drawdown,
            db_path=self.db_path,
        )
        return report, metrics, verdict
