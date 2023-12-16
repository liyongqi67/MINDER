# MINDER Data Processing
## Download pseudo-queries on Wikipedia (NQ and TriviaQA) and MSMARCO.
Please download pseudo-queries for Wikipedia 
## Data processing on NQ and TriviaQA (take TriviaQA as an example)
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
4) Process the data for fairseq training.
```bash
sh preprocess_fairseq.sh data/training_data/Trivia_title_body_query_generated/ data/SEAL/BART_FILES
```
## Data processing on MSMARCO
```bash
python3 scripts/training/make_supervised_msmarco_dataset2.py \
    Tevatron/msmarco-passage /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/dev \
    --target title \
    --mark_target \
    --mark_silver \
    --n_samples 3 \
    --mode a 
python3 scripts/training/make_supervised_msmarco_dataset2.py \
    Tevatron/msmarco-passage /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/dev \
    --target span \
    --mark_target \
    --mark_silver \
    --n_samples 10 \
    --mode a
python3 scripts/training/make_supervised_msmarco_dataset2.py \
    Tevatron/msmarco-passage /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/dev \
    --target query \
    --pid2query /home/v-yongqili/project/GGR/data/MSMARCO/pid2query.pkl \
    --mark_target \
    --mark_silver \
    --n_samples 5 \
    --mode a


# train集合三种表示

python3 scripts/training/make_supervised_msmarco_dataset2.py \
    Tevatron/msmarco-passage /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/train \
    --target title \
    --mark_target \
    --mark_silver \
    --n_samples 3 \
    --mode a
python3 scripts/training/make_supervised_msmarco_dataset2.py \
    Tevatron/msmarco-passage /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/train \
    --target span \
    --mark_target \
    --mark_silver \
    --n_samples 10 \
    --mode a
python3 scripts/training/make_supervised_msmarco_dataset2.py \
    Tevatron/msmarco-passage /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/train \
    --target query \
    --pid2query /home/v-yongqili/project/GGR/data/MSMARCO/pid2query.pkl \
    --mark_target \
    --mark_silver \
    --n_samples 5 \
    --mode a

python3 -u  scripts/training/make_generated_dataset_for_mamarco.py \
    Tevatron/msmarco-passage-corpus \
    /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/unsupervised.source \
    /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/unsupervised.target \
    --format dpr --num_samples 3 --num_title_samples 1 --num_query_samples 5 --full_doc_n 1 --mark_pretraining --pid2query /home/v-yongqili/project/GGR/data/MSMARCO/pid2query.pkl


cat /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/unsupervised.source >> /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/train.source
cat /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/unsupervised.target >> /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2/train.target

sh scripts/training/preprocess_fairseq.sh /home/v-yongqili/project/GGR/data/SEAL/MSMARCO_title_body_query_generated2 /home/v-yongqili/project/GGR/data/SEAL/BART_FILES
```
