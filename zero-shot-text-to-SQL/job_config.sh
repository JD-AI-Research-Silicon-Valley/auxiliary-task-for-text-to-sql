model_config='run.zero-shot-text-to-SQL'
DATANAME='wikisql'
GPU_ID=$1

DATA_DIR=../data_model/$DATANAME    #data, annotated_data, embedding fold are in DATA_DIR
SAVE_PATH=$DATA_DIR/$model_config   #processed data and model are in SAVE_PATH
