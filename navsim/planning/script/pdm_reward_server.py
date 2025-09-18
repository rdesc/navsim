import os
import time
import json
import logging
from typing import List
import numpy as np
from pathlib import Path

from hydra.utils import instantiate
from hydra import initialize, compose
from tqdm import tqdm
from fastapi import FastAPI, Request
from pydantic import BaseModel

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


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("RewardServer")

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

metric_cache_cached = {}
invalid_count = 0
valid_count = 0

# -------------------------------------------------------------------
# Reward function
# -------------------------------------------------------------------
def reward(
    pred_list: List,
    token_list: List,
    debug: bool = True,
) -> List[float]:
    
    global invalid_count, valid_count, metric_cache_cached

    pdm_results = []

    for pred, token in tqdm(zip(pred_list, token_list)):
        valid = True
        start_time = time.time()
        
        if token in metric_cache_cached:
            metric_cache = metric_cache_cached[token]
        else:
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
            pdm_results.append([pdms, valid])
            logger.info(f"Token {token} pdm_score={pdms:.3f}")
            
            if debug:
                logger.debug(f"Token {token}: " + json.dumps(json.loads(score_row_stage_one.iloc[0].to_json()), indent=2))
        else:
            invalid_count += 1
            pdm_results.append([0.0, valid])
            
        logger.info(f"Valid count: {valid_count}, Invalid count: {invalid_count}")
        logger.info(f"Token {token} pdm_score computation took {(time.time() - start_time)*1000:.2f} ms")

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

# @app.post("/reward_batch")
# def compute_reward_batch(batch: BatchInput):
#     rewards = []
#     for item in batch.items:
#         pred = np.array(item.pred, dtype=np.float32)
#         r = reward(pred, item.token, navsim_objs)
#         rewards.append(float(r))
#     logger.info(f"Batch reward computed for {len(batch.items)} items.")
#     return {"rewards": rewards}

# TODO: need to print the predicted trajectories for debugging
# TODO: why do we have so many 0s and 1s?
# TODO: recompute the PDM score with traffic agent enabled in the run_pdm_score_from_submission2
# TODO: what does these trajectories look like for the invalid examples?
# TODO: do we have diversity in the trajectories?
# TODO: check params: temperature = 1.0, top-p of 1.0, and top-k of 0.0