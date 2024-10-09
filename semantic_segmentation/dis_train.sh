MODEL=upernet.biformer_small
OUTPUT_DIR=/home/h3c/workspace/results/biformer/seg

CONFIG_DIR=configs/ade20k
CONFIG=${CONFIG_DIR}/${MODEL}.py

NOW=$(date '+%m-%d-%H:%M:%S')
WORK_DIR=${OUTPUT_DIR}/${MODEL}/${NOW}


python -m torch.distributed.launch --nproc_per_node=2 train.py ${CONFIG} \
            --launcher="pytorch" \
            --work-dir=${WORK_DIR} \

            
# python -u train.py ${CONFIG} \
#             --launcher="none" \
#             --work-dir=${WORK_DIR} \
            