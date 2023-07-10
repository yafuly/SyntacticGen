PROJECT_PATH=""
export PYTHONPATH=$PROJECT_PATH

DATA_PATH="$PROJECT_PATH/toy_data/training_triplets"
MODEL_PATH="$PROJECT_PATH/models"
mkdir -p $MODEL_PATH
fs_plugins="$PROJECT_PATH/fs_plugins"
time=$(date +'%m:%d:%H:%M')

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python3 -u $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH --user-dir $fs_plugins \
	--keep-interval-updates 10 --save-interval-updates 300 --batch-size-valid 256 --validate-interval-updates 300 --maximize-best-checkpoint-metric \
	--eval-bleu-remove-bpe --best-checkpoint-metric bleu --log-format simple --log-interval 100 \
	--eval-bleu --eval-bleu-detok space --keep-last-epochs 1 --keep-best-checkpoints 5  --fixed-validation-seed 7 --ddp-backend=no_c10d \
	--eval-tokenized-bleu \
	--task translation \
	--arch at_tree_attn_nonshare --share-all-embeddings \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
	--lr 0.0007 --stop-min-lr 1e-09 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
	--max-tokens 16000 --max-update 300000 --save-dir $MODEL_PATH \
	--update-freq 1 \
	--eval-bleu-print-samples \
	--ctx-encoder-layers 3 \
	--decoder-layers 3 \
	--dropout 0.3 \
	--fp16 \
	--tensorboard-logdir $MODEL_PATH/tb \
	--patience 40 \
	--dataset-impl 'raw' \
	--save-dir $MODEL_PATH 2>&1 | tee $MODEL_PATH/train.log.$time
 
# remove ''--dataset-impl 'raw' \'' if your data is binarized
