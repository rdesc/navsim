import os
import time
import json
import logging
from typing import List
import numpy as np
from pathlib import Path

from hydra.utils import instantiate
from hydra import initialize, compose
import pandas as pd
from tqdm import tqdm
from fastapi import FastAPI, Request
from pydantic import BaseModel
from cachetools import LRUCache

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.agents.poutine_agent import interp_trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"
TRAIN_TEST_SPLIT = "navtrain"
CACHE_PATH = f"{os.getenv('NAVSIM_EXP_ROOT')}/metric_cache_{TRAIN_TEST_SPLIT}"
DEBUG = True


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("PDMSRewardServer")

# -------------------------------------------------------------------
# NAVSIM Object instantiation
# -------------------------------------------------------------------
with initialize(config_path=CONFIG_PATH, version_base=None):
    # compose full config including _base_ hierarchy + CLI overrides
    cfg = compose(config_name=CONFIG_NAME)

cfg.metric_cache_path = CACHE_PATH
cfg.train_test_split = TRAIN_TEST_SPLIT

# Subscore weights
subscore_weights = cfg.scorer.config
subscore_weights.progress_weight = 5.0                    # default 5.0
subscore_weights.ttc_weight = 5.0                         # default 5.0
subscore_weights.lane_keeping_weight = 2.0                # default 2.0
subscore_weights.history_comfort_weight = 2.0             # default 2.0
subscore_weights.two_frame_extended_comfort_weight = 2.0  # default 2.0
subscore_weights.human_penalty_filter = False  # NOTE: this is broken in the e2e challenge code, so disable, right??

# custom soft weights for penalty terms
no_at_fault_collisions_weight = 1e-1
drivable_area_compliance_weight = 1e-1
driving_direction_compliance_weight = 1e-1
traffic_light_compliance_weight = 1e-1

metric_cache_path = Path(cfg.metric_cache_path)
metric_cache_loader = MetricCacheLoader(metric_cache_path)
simulator: PDMSimulator = instantiate(cfg.simulator)
scorer: PDMScorer = instantiate(cfg.scorer)
traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
    cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
)

assert (
    simulator.proposal_sampling == scorer.proposal_sampling
), "Simulator and scorer proposal sampling has to be identical"

metric_cache_cached = LRUCache(maxsize=1000)
invalid_count = 0
valid_count = 0
running_average_pdms = 0.0
running_average_pdms_valid = 0.0


def compute_soft_pdms(df: pd.DataFrame) -> float:
    """Compute a soft PDM score using custom weights for penalty terms."""
    # Extract individual metrics from the DataFrame
    ego_progress = df["ego_progress"].item()
    time_to_collision_within_bound = df["time_to_collision_within_bound"].item()
    lane_keeping = df["lane_keeping"].item()
    history_comfort = df["history_comfort"].item()
    
    no_at_fault_collisions = df["no_at_fault_collisions"].item()
    drivable_area_compliance = df["drivable_area_compliance"].item()
    driving_direction_compliance = df["driving_direction_compliance"].item()
    traffic_light_compliance = df["traffic_light_compliance"].item()

    # Compute weighted metrics
    weighted_metrics = (
        ego_progress * subscore_weights.progress_weight +
        time_to_collision_within_bound * subscore_weights.ttc_weight +
        lane_keeping * subscore_weights.lane_keeping_weight +
        history_comfort * subscore_weights.history_comfort_weight
    ) / (
        subscore_weights.progress_weight +
        subscore_weights.ttc_weight +
        subscore_weights.lane_keeping_weight +
        subscore_weights.history_comfort_weight
    )  # normalize to sum to 1
    
    # Compute penalty metrics with custom weights
    penalty_metrics = (
        (no_at_fault_collisions or no_at_fault_collisions_weight) *
        (drivable_area_compliance or drivable_area_compliance_weight) *
        (driving_direction_compliance or driving_direction_compliance_weight) *
        (traffic_light_compliance or traffic_light_compliance_weight)
    )

    # Compute soft PDM score as the average of weighted metrics
    soft_pdms = weighted_metrics * penalty_metrics

    return soft_pdms

# -------------------------------------------------------------------
# Reward function
# -------------------------------------------------------------------
def reward(
    pred_list: List,
    token_list: List,
) -> List[float]:

    global invalid_count, valid_count, metric_cache_cached, running_average_pdms, running_average_pdms_valid

    pdm_results = []

    for pred, token in tqdm(zip(pred_list, token_list)):
        valid = True
        start_time = time.time()
        
        if token in metric_cache_cached:
            metric_cache = metric_cache_cached[token]
            logger.debug(f"Metric cache hit for token={token}")
        else:
            logger.debug(f"Metric cache miss for token={token}")
            metric_cache = metric_cache_loader.get_from_token(token)  # NOTE: this is the part that is slow!!
            metric_cache_cached[token] = metric_cache

        try:
            assert np.asarray(pred).shape == (5, 2), f"Expected pred shape (5, 2), got {np.asarray(pred).shape}"
            pred = interp_trajectory(pred, 1.0, 0.25, 4.0, compute_heading=True)
            assert pred.shape == (16, 3), f"Expected pred shape (16, 3), got {pred.shape}"
        
            score_row_stage_one, _ = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(pred, TrajectorySampling(time_horizon=4, interval_length=0.25)),
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            pdms = float(score_row_stage_one["pdm_score"].item())
            
            # confirm pdms is within valid range
            if not (0.0 <= pdms <= 1.0):
                valid = False
                logger.warning(f"Invalid PDM score range for token={token}: {pdms}")
    
        except Exception as e:
            valid = False
            logger.warning(f"Error occurred while computing PDM score for token={token}: {e}")

        if valid:
            valid_count += 1
            pdms = float(score_row_stage_one["pdm_score"].item())
            
            soft_pdms = compute_soft_pdms(score_row_stage_one)
            
            pdm_results.append([soft_pdms, valid])
            logger.info(f"Token {token} pdm_score={pdms:.3f}, soft_pdms={soft_pdms:.3f}")
            
            logger.debug(f"Token {token}: " + json.dumps(json.loads(score_row_stage_one.iloc[0].to_json()), indent=2))
        else:
            invalid_count += 1
            pdm_results.append([0.0, valid])
            
        logger.info(f"Valid count: {valid_count}, Invalid count: {invalid_count}")
        logger.info(f"Token {token} pdm_score computation took {(time.time() - start_time)*1000:.2f} ms")

    curr_avg = np.mean([r for r, v in pdm_results])
    curr_avg_valid = np.mean([r for r, v in pdm_results if v])
    total_count = valid_count + invalid_count
    running_average_pdms = (running_average_pdms * (total_count - 1) + curr_avg * len(pdm_results)) / total_count
    running_average_pdms_valid = (running_average_pdms_valid * (valid_count - 1) + curr_avg_valid * len([r for r, v in pdm_results if v])) / valid_count if valid_count > 0 else 0.0

    logger.info(f"Running average PDMS: {running_average_pdms:.3f}")
    logger.info(f"Running average PDMS (valid only): {running_average_pdms_valid:.3f}")

    return pdm_results


# -------------------------------------------------------------------
# FastAPI app
#
# launch with:
# uvicorn navsim.planning.script.pdm_reward_server:app --host 0.0.0.0 --port 8000
#
# -------------------------------------------------------------------
app = FastAPI()

class TrajectoryInput(BaseModel):
    pred: List[List[float]]
    token: str

class BatchInput(BaseModel):
    items: List[TrajectoryInput]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log latency for every request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f} ms")
    return response

@app.post("/reward")
def compute_reward(item: TrajectoryInput):
    logger.info(f"Received reward request for token={item.token} and pred={item.pred}")
    pred = np.array(item.pred, dtype=np.float32)
    token_id = item.token
    pred_list = [pred]
    token_list = [token_id]
    r, valid = reward(pred_list, token_list)[0]
    
    if not valid:
        logger.warning(f"Invalid reward computation for token={token_id} with {pred_list}. Returning reward=0.0")
        r = 0.0
    else:
        logger.info(f"Single reward computed for token={token_id}, reward={float(r):.3f}")
        
    return {"reward": float(r), "valid": valid}
