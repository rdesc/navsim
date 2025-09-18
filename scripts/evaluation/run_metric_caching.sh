for TRAIN_TEST_SPLIT in warmup_two_stage navhard_two_stage navtrain; do
    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
        train_test_split=$TRAIN_TEST_SPLIT \
        metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_$TRAIN_TEST_SPLIT
done
