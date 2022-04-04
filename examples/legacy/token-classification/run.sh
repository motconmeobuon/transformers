## The relevant files are currently on a shared Google
## drive at https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J
## Monitor for changes and eventually migrate to nlp dataset
mkdir -p villa_ner

curl -L 'https://raw.githubusercontent.com/motconmeobuon/villaaaa/master/train.conll'  | tr '\t' ' ' > villa_ner/train.txt.tmp
curl -L 'https://raw.githubusercontent.com/motconmeobuon/villaaaa/master/val.conll' | tr '\t' ' ' > villa_ner/dev.txt.tmp
curl -L 'https://raw.githubusercontent.com/motconmeobuon/villaaaa/master/test.conll' | tr '\t' ' ' > villa_ner/test.txt.tmp

export MAX_LENGTH=64
export BERT_MODEL=vinai/phobert-base
python3 scripts/preprocess.py villa_ner/train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 scripts/preprocess.py villa_ner/dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python3 scripts/preprocess.py villa_ner/test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
export OUTPUT_DIR=germeval-model
export BATCH_SIZE=16
export NUM_EPOCHS=75
export SAVE_STEPS=20000
export SEED=1

python3 run_ner.py \
--task_type NER \
--data_dir . \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
