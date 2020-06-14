#!/bin/bash

# variables

GPU_INDEX=2 # which GPU to run on
DATA_SOURCE_PREFIX_DIR="../../../unborn/shared/sayeri/iqa_data_source/" # update accordingly
DATA_SOURCE_DIR="reorganized_combined_dataset" #update accordingly
DATA_SOURCE_DIR=$DATA_SOURCE_PREFIX_DIR$DATA_SOURCE_DIR

# TO DO: hyperparameters for the model training
DEBUG_MODE=0 # 0: run time debug 1: tune lr 2: full training

MODEL_TYPE="vgg_16_transfer" #shallow, vgg_11, vgg_16_scratch, vgg_16_transfer

ADJUST_LR_FLAG="0"
LOAD_MODEL_FLAG="0"
DECAY="0.0"
learning_rates=("0.000025")

if [ $DEBUG_MODE ==  0 ] # run time debug
then
	BATCH_SIZE="3" # 9 instances in the set
	EPOCHS="2"
	DATA_PARTITION_DIR="dataset_partition/debug"
	DATA_PARTITION_DIR=$DATA_SOURCE_PREFIX_DIR$DATA_PARTITION_DIR
	SAVE_IMAGES="1" # debugging augmentations
	SAVE_MODEL="1"
	ENABLE_AUGMENTATIONS="1"
elif [ $DEBUG_MODE ==  1 ] #tune lr
then
	BATCH_SIZE="50" # 20 for vgg dual
	EPOCHS="2000"
	DATA_PARTITION_DIR="dataset_partition/tune_lr"
	DATA_PARTITION_DIR=$DATA_SOURCE_PREFIX_DIR$DATA_PARTITION_DIR
	SAVE_IMAGES="0" 
	SAVE_MODEL="0"
	ENABLE_AUGMENTATIONS="1"
	learning_rates=("1e-2" "1e-3" "1e-4" "1e-5" "1e-6")
	learning_rates=("0.000025")
else # train
	BATCH_SIZE="50" # 20 for vgg dual
	EPOCHS="1000"
	DATA_PARTITION_DIR="dataset_partition/complete"
	DATA_PARTITION_DIR=$DATA_SOURCE_PREFIX_DIR$DATA_PARTITION_DIR
	SAVE_IMAGES="0" 
	SAVE_MODEL="1"
	ENABLE_AUGMENTATIONS="1"
fi


for curr_lr in "${learning_rates[@]}"
do
	echo $curr_lr
	
	MAIN_RES_PATH="$GPU_INDEX"
	tmp="_"
	MAIN_RES_PATH="$MAIN_RES_PATH$tmp"
	MAIN_RES_PATH="$MAIN_RES_PATH$MODEL_TYPE"
	tmp="_lr_"
	MAIN_RES_PATH="$MAIN_RES_PATH$tmp"
	MAIN_RES_PATH="$MAIN_RES_PATH$curr_lr"

	stderr_output_path="$GPU_INDEX"
	tmp="_stderr_train_model.txt"
	stderr_output_path="$stderr_output_path$tmp"

	stdout_output_path="$GPU_INDEX"
	tmp="_stdout_train_model.txt"
	stdout_output_path="$stdout_output_path$tmp"

	CUDA_VISIBLE_DEVICES=$GPU_INDEX python2 train_model.py \
	--save_images $SAVE_IMAGES --save_model $SAVE_MODEL \
	--enable_augmentations $ENABLE_AUGMENTATIONS \
	--model_type $MODEL_TYPE \
	--lr $curr_lr --decay $DECAY --batch_size $BATCH_SIZE --epochs $EPOCHS \
	--adjust_lr_flag $ADJUST_LR_FLAG \
	--load_model_flag $LOAD_MODEL_FLAG \
	--data_source_dir $DATA_SOURCE_DIR \
	--data_partition_dir $DATA_PARTITION_DIR \
	--main_res_path $MAIN_RES_PATH \
	2> $stderr_output_path | tee $stdout_output_path

	if [ $DEBUG_MODE ==  1 ] #tune lr
	then
		continue
	fi

	stderr_output_path="$GPU_INDEX"
	tmp="_stderr_eval_model.txt"
	stderr_output_path="$stderr_output_path$tmp"

	stdout_output_path="$GPU_INDEX"
	tmp="_stdout_eval_model.txt"
	stdout_output_path="$stdout_output_path$tmp"

	CUDA_VISIBLE_DEVICES=$GPU_INDEX python2 eval_model.py \
	--model_type $MODEL_TYPE \
	--data_source_dir $DATA_SOURCE_DIR \
	--data_partition_dir $DATA_PARTITION_DIR \
	--main_res_path $MAIN_RES_PATH \
	--save_images $SAVE_IMAGES \
	2> $stderr_output_path | tee $stdout_output_path
	
done 

if [ $DEBUG_MODE ==  1 ] #tune lr
then
	stderr_output_path="$GPU_INDEX"
	tmp="_stderr_plot_learning_curves.txt"
	stderr_output_path="$stderr_output_path$tmp"

	stdout_output_path="$GPU_INDEX"
	tmp="_stderr_plot_learning_curves.txt"
	stdout_output_path="$stdout_output_path$tmp"

	learning_rates="1e-2 1e-3 1e-4 1e-5 1e-6"
	# learning_rates="1e-2 1e-3"
	RES_PATH="."


	LR_PATH_PREFIX="$GPU_INDEX"
	tmp="_"
	LR_PATH_PREFIX="$LR_PATH_PREFIX$tmp"
	LR_PATH_PREFIX="$LR_PATH_PREFIX$DATA_TYPE"
	LR_PATH_PREFIX="$LR_PATH_PREFIX$tmp"
	LR_PATH_PREFIX="$LR_PATH_PREFIX$MODEL_TYPE"
	tmp="_lr"
	LR_PATH_PREFIX="$LR_PATH_PREFIX$tmp"
	LR_PATH_PREFIX="$LR_PATH_PREFIX$curr_lr"


	SAVE_PATH_PREFIX="$GPU_INDEX"
	tmp="_"
	SAVE_PATH_PREFIX="$SAVE_PATH_PREFIX$tmp"
	SAVE_PATH_PREFIX="$SAVE_PATH_PREFIX$DATA_TYPE"


	python2 plot_learning_curves.py --learning_rates $learning_rates --res_path $RES_PATH \
	--lr_path_prefix $LR_PATH_PREFIX --save_path_prefix $SAVE_PATH_PREFIX \
	2> $stderr_output_path | tee $stdout_output_path	
fi




# sudo /sbin/shutdown -h now
