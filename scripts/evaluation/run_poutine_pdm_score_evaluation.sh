# Usage: ./run_poutine_pdm_score_evaluation.sh <train_test_split> <path/to/preds_dict.pkl> <experiment_name>
# Example: ./run_eval.sh warmup_two_stage /path/to/preds_dict.pkl navsim_eval_no-cot_warmup_two_stage

if [ $# -lt 3 ]; then
  echo "Usage: $0 <train_test_split> <load_predictions_from_file> <experiment_name>"
  exit 1
fi

TRAIN_TEST_SPLIT=$1  # warmup_two_stage, navhard_two_stage or navtrain
LOAD_PREDICTIONS_FROM_FILE=$2
EXPERIMENT_NAME=$3

CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_$TRAIN_TEST_SPLIT
SYNTHETIC_SENSOR_PATH="$OPENSCENE_DATA_ROOT/$TRAIN_TEST_SPLIT/sensor_blobs"
SYNTHETIC_SCENES_PATH="$OPENSCENE_DATA_ROOT/$TRAIN_TEST_SPLIT/synthetic_scene_pickles"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  "agent=poutine_agent" \
  agent.load_predictions_from_file="$LOAD_PREDICTIONS_FROM_FILE" \
  experiment_name=${EXPERIMENT_NAME}_${TRAIN_TEST_SPLIT} \
  metric_cache_path=$CACHE_PATH \
  synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
  synthetic_scenes_path=$SYNTHETIC_SCENES_PATH

# NOTE: run_pdm_score_from_submission.py does not have distributed evaluation so way slower!
# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py \
#   train_test_split=$TRAIN_TEST_SPLIT \
#   submission_file_path=$SUBMISSION_FILE_PATH \
#   experiment_name=${EXPERIMENT_NAME}_${TRAIN_TEST_SPLIT} \
#   metric_cache_path=$CACHE_PATH \
#   synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
#   synthetic_scenes_path=$SYNTHETIC_SCENES_PATH