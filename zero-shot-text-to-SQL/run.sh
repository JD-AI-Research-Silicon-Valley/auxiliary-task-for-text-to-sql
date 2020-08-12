. ./job_config.sh

#MODEL_PATH=$(head -n1 $SAVE_PATH/dev_best.txt)
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py -unseen_table 'attpool_vis' -split dev  -data_path "$DATA_DIR" -save_path "$SAVE_PATH" -model_path "$MODEL_PATH"

python preprocess.py -train_anno "$DATA_DIR/annotated_ent/train.jsonl" -valid_anno "$DATA_DIR/annotated_ent/dev.jsonl" -test_anno "$DATA_DIR/annotated_ent/test.jsonl" -save_data "$SAVE_PATH"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py -start_checkpoint_at 25 -split_type "incell" -epochs 45 -global_attention "general" -fix_word_vecs -dropout 0.5 -score_size 64 -attn_hidden 64 -rnn_size 250 -co_attention -embd "$DATA_DIR" -data "$SAVE_PATH" -save_dir "$SAVE_PATH"   >out-train

CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py -unseen_table 'full' -split dev  -data_path "$DATA_DIR" -save_path "$SAVE_PATH" -model_path "$SAVE_PATH/m_*.pt"  >out-dev
MODEL_PATH=$(head -n1 $SAVE_PATH/dev_best.txt)
echo $MODEL_PATH
CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py -unseen_table 'full' -split finaltest -data_path "$DATA_DIR" -save_path "$SAVE_PATH" -model_path "$MODEL_PATH"  >out-test

CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py -unseen_table 'zs' -split finaltest -data_path "$DATA_DIR" -save_path "$SAVE_PATH" -model_path "$MODEL_PATH"  >out-test-zs




