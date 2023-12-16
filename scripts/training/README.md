# MINDER Data Processing
1) Run scripts to create training and validation examples.
```bash
python3 scripts/training/make_supervised_dpr_dataset.py \
    data/Trivia/biencoder-trivia-dev.json data/training_data/Trivia_title_body_query_generated/dev \
    --target title \
    --mark_target \
    --mark_silver \
    --n_samples 3 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python3 scripts/training/make_supervised_dpr_dataset.py \
    data/Trivia/biencoder-trivia-dev.json data/training_data/Trivia_title_body_query_generated/dev \
    --target span \
    --mark_target \
    --mark_silver \
    --n_samples 10 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python3 scripts/training/make_supervised_dpr_dataset.py \
    data/Trivia/biencoder-trivia-dev.json /home/v-yongqili/project/GGR/data/training_data/Trivia_title_body_query_generated/dev \
    --target query \
    --pid2query /home/v-yongqili/project/GGR/data/pid2query.pkl \
    --mark_target \
    --mark_silver \
    --n_samples 5 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

# train
python3 scripts/training/make_supervised_dpr_dataset.py \
    data/Trivia/biencoder-trivia-train.json data/training_data/Trivia_title_body_query_generated/train \
    --target title \
    --mark_target \
    --mark_silver \
    --n_samples 3 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python3 scripts/training/make_supervised_dpr_dataset.py \
    data/Trivia/biencoder-trivia-train.json data/training_data/Trivia_title_body_query_generated/train \
    --target span \
    --mark_target \
    --mark_silver \
    --n_samples 10 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python3 scripts/training/make_supervised_dpr_dataset.py \
    data/Trivia/biencoder-trivia-train.json data/training_data/Trivia_title_body_query_generated/train \
    --target query \
    --pid2query /home/v-yongqili/project/GGR/data/pid2query.pkl \
    --mark_target \
    --mark_silver \
    --n_samples 5 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0
```
2) Add unsupervised examples.
```bash
   python3 -u  scripts/training/make_generated_dataset2.py \
    data/psgs_w100.tsv \
    data/training_data/Trivia_title_body_query_generated/unsupervised.source \
    data/training_data/Trivia_title_body_query_generated/unsupervised.target \
    --format dpr --num_samples 3 --num_title_samples 1 --num_query_samples 2 --full_doc_n 1 --mark_pretraining --pid2query /home/v-yongqili/project/GGR/data/pid2query.pkl
```
3) Merge supervised and unsupervised examples
```bash
cat data/training_data/Trivia_title_body_query_generated/unsupervised.source >> data/training_data/Trivia_title_body_query_generated/train.source
cat data/training_data/Trivia_title_body_query_generated/unsupervised.target >> data/training_data/Trivia_title_body_query_generated/train.target
```
