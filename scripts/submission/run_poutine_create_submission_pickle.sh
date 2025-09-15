#!/bin/bash
# Usage: ./run_eval_private.sh <submission_file_path_or_empty> <experiment_name_or_empty>
# Example (step 1: no predictions, cache examples to file):
#   ./run_eval_private.sh 
# Example (step 2: with predictions):
#   ./run_eval_private.sh /path/to/preds_dict.pkl navsim_eval_private

TRAIN_TEST_SPLIT=private_test_hard_two_stage
LOAD_PREDICTIONS_FROM_FILE=${1:-""}
EXPERIMENT_NAME=${2:-""}

TEAM_NAME=''
AUTHORS='Rodrigue de Schaetzen, Luke Rowe, Roger Girgis, Christopher Pal, Liam Paull'
EMAIL='rodrigue.deschaetzen@mila.quebec'
INSTITUTION='Mila - Quebec AI Institute, Universite de Montreal, Polytechnique Montreal, CIFAR AI Chair'
COUNTRY='Canada'

EVAL_MODE=test
SYNTHETIC_SENSOR_PATH="$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/sensor_blobs"
SYNTHETIC_SCENES_PATH="$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/openscene_meta_datas"

# poutine agent args
ORIGINAL_SENSOR_PATH="$OPENSCENE_DATA_ROOT/sensor_blobs/private_test_hard"
if [ -n "$LOAD_PREDICTIONS_FROM_FILE" ]; then
  CACHE_DATASET_TO_FILE=""
  EXPERIMENT_NAME=${EXPERIMENT_NAME}_${TRAIN_TEST_SPLIT}
else
  CACHE_DATASET_TO_FILE="$DATASET_ROOT/poutine_processed_navsim/dataset_pickles/ego_status_dataset_navsim_${EVAL_MODE}_split_${TRAIN_TEST_SPLIT}.json"
  EXPERIMENT_NAME=submission_poutine_agent_dummy
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle_challenge.py" \
  "train_test_split=$TRAIN_TEST_SPLIT" \
  "agent=poutine_agent" \
  agent.jpeg_root_paths="[${ORIGINAL_SENSOR_PATH}, ${SYNTHETIC_SENSOR_PATH}]" \
  agent.cache_dataset_to_file="$CACHE_DATASET_TO_FILE" \
  agent.load_predictions_from_file="$LOAD_PREDICTIONS_FROM_FILE" \
  "experiment_name=$EXPERIMENT_NAME" \
  "team_name='$TEAM_NAME'" \
  "authors='$AUTHORS'" \
  "email='$EMAIL'" \
  "institution='$INSTITUTION'" \
  "country='$COUNTRY'" \
  "synthetic_sensor_path='$SYNTHETIC_SENSOR_PATH'" \
  "synthetic_scenes_path='$SYNTHETIC_SCENES_PATH'"
  
if ! [ -n "$LOAD_PREDICTIONS_FROM_FILE" ]; then
  echo -e "\nDataset was cached to file: $CACHE_DATASET_TO_FILE"
fi