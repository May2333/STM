# export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='ft_mc_msrvtt'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=8
NUM_CPU=112

torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks/retrieval_mc.py \
    $(dirname $0)/b16_5m.py \
    pretrained_path  /YOUR_PATH/projects/stm/multi_modality/exp/pretraining/output/10epoch_umt_invreconstrucedloss07_actionclip_allila035do01temproalonly_attnwith1-05lambdatemporal1continuewith05lambda005_teachervdofea_tau005_visvtm_alltokenutaplustext_multiilalayer/ckpt_latest.pth \
    output_dir ${OUTPUT_DIR}
