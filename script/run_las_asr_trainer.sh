"""
ClovaCall
Copyright 2020-present NAVER Corp.
MIT license
"""
#!/bin/bash

MAIN_DIR=${0%/*}
cd $MAIN_DIR/..

TARGET_CODE=las.pytorch/main.py
MODEL_PATH=models
LOG_PARENT_PATH=log

if [ ! -f $TARGET_CODE ]; then
    echo "[ERROR] TARGET_CODE($TARGET_CODE) not found."
    exit
fi

if [ ! -d $MODEL_PATH ]; then
    mkdir $MODEL_PATH
fi

if [ ! -d $LOG_PARENT_PATH ]; then
    mkdir $LOG_PARENT_PATH
fi

################################################################
##	Careful while modifying lines above.
################################################################

TRAIN_FILE=data/ClovaCall/train_ClovaCall.json
TEST_FILE=data/ClovaCall/test_ClovaCall.json
LABEL_FILE=data/kor_syllable.json
DATASET_PATH=data/ClovaCall/clean

CUDA_DEVICE_ID=0

# Default
RNN_TYPE=LSTM
BATCH_SIZE=32
LR=3e-4
LR_ANNEAL=1.1
DROPOUT=0.3
TF_RATIO=1.0
EPOCHS=100

# LAS
ENCODER_LAYERS=3
ENCODER_SIZE=512
DECODER_LAYERS=2
DECODER_SIZE=512

GPU_SIZE=1
CPU_SIZE=4


TRAIN_INFO="ClovaCall"

################################################################
##	Careful while modifying lines below.
################################################################

MODELS_PATH=models/$TRAIN_INFO


CUR_MODEL_PATH=${MODELS_PATH}/${RNN_TYPE}_${ENCODER_SIZE}x${ENCODER_LAYERS}_${DECODER_SIZE}x${DECODER_LAYERS}_${TRAIN_INFO}
LOG_CHILD_PATH=${LOG_PARENT_PATH}/${RNN_TYPE}_${ENCODER_SIZE}x${ENCODER_LAYERS}_${DECODER_SIZE}x${DECODER_LAYERS}_${TRAIN_INFO}

LOG_FILE=$LOG_CHILD_PATH/run_las_asr_trainer_CUDA${CUDA_DEVICE_ID}.sh.log

if [ ! -d $MODELS_PATH ]; then
    mkdir $MODELS_PATH
fi

if [ ! -d $CUR_MODEL_PATH ]; then
    mkdir $CUR_MODEL_PATH
fi

if [ ! -d $LOG_CHILD_PATH ]; then
    mkdir $LOG_CHILD_PATH
fi


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID \
python -u $TARGET_CODE \
--batch_size $BATCH_SIZE \
--num_workers $CPU_SIZE \
--num_gpu $GPU_SIZE \
--rnn-type $RNN_TYPE \
--lr $LR \
--learning-anneal $LR_ANNEAL \
--dropout $DROPOUT \
--teacher_forcing $TF_RATIO \
--encoder_layers $ENCODER_LAYERS --encoder_size $ENCODER_SIZE \
--decoder_layers $DECODER_LAYERS --decoder_size $DECODER_SIZE \
--train-file $TRAIN_FILE --test-file-list $TEST_FILE \
--labels-path $LABEL_FILE \
--dataset-path $DATASET_PATH \
--cuda --save-folder $CUR_MODEL_PATH --model-path $CUR_MODEL_PATH/final.pth --log-path $LOG_CHILD_PATH | tee $LOG_FILE 

# If you want to continue training from previous model or finetuning, add these commands.
# --load-model
# --finetune
