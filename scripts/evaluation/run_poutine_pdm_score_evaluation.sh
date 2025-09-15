# Usage: ./run_eval.sh train_test_split path/to/submission.pkl experiment_name
# Example: ./run_eval.sh warmup_two_stage /path/to/preds_dict.pkl navsim_eval_no-cot_warmup_two_stage

if [ $# -lt 3 ]; then
  echo "Usage: $0 <train_test_split> <submission_file_path> <experiment_name>"
  exit 1
fi

TRAIN_TEST_SPLIT=$1  # warmup_two_stage or navhard_two_stage
SUBMISSION_FILE_PATH=$2
EXPERIMENT_NAME=$3

CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH="$OPENSCENE_DATA_ROOT/$TRAIN_TEST_SPLIT/sensor_blobs"
SYNTHETIC_SCENES_PATH="$OPENSCENE_DATA_ROOT/$TRAIN_TEST_SPLIT/synthetic_scene_pickles"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  submission_file_path=$SUBMISSION_FILE_PATH \
  experiment_name=${EXPERIMENT_NAME}_${TRAIN_TEST_SPLIT} \
  metric_cache_path=$CACHE_PATH \
  synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
  synthetic_scenes_path=$SYNTHETIC_SCENES_PATH

# /home/jovyan/workspace/navsim_workspace/exp/submission_poutine_agent_warmup_two_stage/2025.09.12.12.56.22/submission.pkl
# /home/jovyan/workspace/navsim_workspace/exp/submission_poutine_agent_warmup_two_stage/2025.09.12.13.16.23/submission.pkl

# /home/jovyan/shared/RodDeSc/experiments/qwen2_5vl_full_sft_navsim_cot_waymo_val_poutine_1300_ckpt_200/navsim_eval_no-cot_warmup_two_stage/preds_dict.pkl: 
# /home/jovyan/workspace/navsim_workspace/exp/submission_poutine_agent_warmup_two_stage/2025.09.12.16.24.59/submission.pkl

# /home/jovyan/shared/RodDeSc/experiments/qwen2_5vl_full_sft_navsim_cot_waymo_val_poutine_1300/navsim_eval_no-cot_warmup_two_stage/preds_dict_20250912_182943.pkl
# /home/jovyan/workspace/navsim_workspace/exp/submission_poutine_agent_warmup_two_stage/2025.09.12.18.32.25/submission.pkl