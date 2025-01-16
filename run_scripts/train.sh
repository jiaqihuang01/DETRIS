dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="DETRIS_base.yaml"
gpu="0,1"
np=$(echo $gpu | tr -cd ',' | wc -c)
np=$((np + 1))
omp=8

filename=$dataset_name"_$(date +%m%d_%H%M%S)"


CUDA_VISIBLE_DEVICES=$gpu \
OMP_NUM_THREADS=$omp \
MASTER_PORT=29560 \
torchrun --nproc_per_node=$np --master_port=29999 \
train.py \
--config config/$dataset_name/$config_name \


