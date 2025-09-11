TEAM_NAME=''
AUTHORS='Rodrigue de Schaetzen, Luke Rowe, Roger Girgis, Christopher Pal, Liam Paull'
EMAIL='rodrigue.deschaetzen@mila.quebec'
INSTITUTION='Mila - Quebec AI Institute, Universite de Montreal, Polytechnique Montreal, CIFAR AI Chair'
COUNTRY='Canada'

TRAIN_TEST_SPLIT=private_test_hard_two_stage
EVAL_MODE=test
SYNTHETIC_SENSOR_PATH="$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/sensor_blobs"
SYNTHETIC_SCENES_PATH="$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/openscene_meta_datas"

EXPERIMENT_NAME=submission_poutine_agent_dummy

# poutine agent args
ORIGINAL_SENSOR_PATH="$OPENSCENE_DATA_ROOT/sensor_blobs/test"
CACHE_DATASET_TO_FILE="$DATASET_ROOT/poutine_processed_navsim/dataset_pickles/ego_status_dataset_navsim_${EVAL_MODE}_split_${TRAIN_TEST_SPLIT}.json"
LOAD_PREDICTIONS_FROM_FILE=""

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
  