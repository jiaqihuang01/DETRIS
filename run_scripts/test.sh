dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="DETRIS_base.yaml"
gpu=0
split_name="val" # "val", "testA", "testB" 1
model_path=''
# Evaluation on the specified of the specified dataset
filename=$dataset_name"_$(date +%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=$gpu \
python3 \
-u test.py \
--config config/$dataset_name/$config_name \
--path $model_path \
--opts TEST.test_split $split_name \
            TEST.test_lmdb datasets/lmdb/$dataset_name/$split_name.lmdb

