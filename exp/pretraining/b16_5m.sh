#export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='output/STM'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=8
NUM_CPU=112

    
# CUDA_VISIBLE_DEVICES=4,5,6,7 
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    tasks/pretrain.py \
    $(dirname $0)/b16_5m.py \
    output_dir ${OUTPUT_DIR}