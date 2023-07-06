PROJECT_PATH=""
DATA_PATH="$PROJECT_PATH/toy_data"
python3 $PROJECT_PATH/prepare_training_data.py src tgt $DATA_PATH/training/train.src $DATA_PATH/training/train.tgt.merge.parse $DATA_PATH/training_triplets

# For toy example, we simply duplicate train set to valid and test set for sanity check.
# Replace with the true files for training.
BPE_SCRIPTS=$PROJECT_PATH/shell/apply-bpe.sh
BPE_CODE="$DATA_PATH/code.6000"
for split in train valid test; do
    for lang in src tgt; do
    cat $DATA_PATH/training_triplets/train.$lang | bash $BPE_SCRIPTS $BPE_CODE  > $DATA_PATH/training_triplets/$split.src-tgt.$lang
    done
done
