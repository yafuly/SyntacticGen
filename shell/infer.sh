PROJECT_PATH=""
export PYTHONPATH=$PROJECT_PATH

DATA_PATH="$PROJECT_PATH/toy_data/inference"
MODEL_PATH="$PROJECT_PATH/models"
mkdir -p $MODEL_PATH
fs_plugins="$PROJECT_PATH/fs_plugins"
time=$(date +'%m:%d:%H:%M')


gpu=0
test_model=avg_best.pt
subset=test
max_iter=30
beam=5
CUDA_VISIBLE_DEVICES=$gpu python3 $PROJECT_PATH/fairseq_cli/generate.py \
    $DATA_PATH \
    --user-dir $fs_plugins \
    --gen-subset $subset \
    --dataset-impl 'raw' \
    --task translation_syntactic_gen \
    --generator 'SyntacticGenerator' \
    --cur-score-ratio 0.2 \
    --prev-score-ratio 0.8 \
    --path $MODEL_PATH/$test_model  \
    --remove-bpe \
    --beam $beam \
    --max-iter $max_iter \
    --max-tokens 8000 \
    --batch-size 512 \
    --fp16 2>&1 | tee $MODEL_PATH/test.$test_model.$subset.B${beam}.iter${max_iter}.log.$time


