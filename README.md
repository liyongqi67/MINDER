# MINDER
This is the official implementation for the paper "Multiview Identifiers Enhanced Generative Retrieval".
# Install
```commandline
git clone https://github.com/liyongqi67/MINDER.git
sudo apt install swig
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -r requirements.txt
pip install -e .
```
# Data
Please download all the data into the `data` folder.
1) `data/NQ` folder. Please download `biencoder-nq-dev.json, biencoder-nq-train.json, nq-dev.qa.csv, nq-test.qa.csv` files into the `NQ` folder from the DPR repositiory (https://github.com/facebookresearch/DPR).
2) `data/Trivia` folder. Please download `biencoder-trivia-dev.json, biencoder-trivia-train.json, trivia-dev.qa.csv, trivia-test.qa.csv` files into the `Trivia` folder from the DPR repositiory (https://github.com/facebookresearch/DPR).
3) `data/fm_index/` folder. Please download fm_index files `psgs_w100.fm_index.fmi, psgs_w100.fm_index.oth` for the Wikipedia corpus and  `msmarco-passage-corpus.fm_index.fmi, msmarco-passage-corpus.fm_index.oth` for the MSMARCO corpus [https://drive.google.com/drive/folders/1xFs9rF49v3A-HODGp6W7mQ4hlo5FspnV?usp=sharing].
4) `data/training_data/NQ_title_body_query_generated` folder. Please download the training files from this link [https://drive.google.com/drive/folders/1Mb5fj8MSQ8djMvAVvYjUiY41Wrk0oiCh?usp=sharing].

# Model training
We use the fairseq to train the BART_large model with the translation task. The script is 
```bash
    - fairseq-train
        data/training_data/NQ_title_body_query_generated/bin 
        --finetune-from-model /bart.large/model.pt 
        --arch bart_large 
        --task translation 
        --criterion label_smoothed_cross_entropy 
        --source-lang source 
        --target-lang target 
        --truncate-source 
        --label-smoothing 0.1 
        --max-tokens 4096 
        --update-freq 1 
        --max-update 800000 
        --required-batch-size-multiple 1
        --validate-interval 1000000
        --save-interval 1000000
        --save-interval-updates 15000 
        --keep-interval-updates 3 
        --dropout 0.1 
        --attention-dropout 0.1 
        --relu-dropout 0.0 
        --weight-decay 0.01 
        --optimizer adam 
        --adam-betas "(0.9, 0.999)" 
        --adam-eps 1e-08 
        --clip-norm 0.1 
        --lr-scheduler polynomial_decay 
        --lr 3e-05 
        --total-num-update 800000 
        --warmup-updates 500 
        --fp16 
        --num-workers 10 
        --no-epoch-checkpoints 
        --share-all-embeddings 
        --layernorm-embedding 
        --share-decoder-input-output-embed 
        --skip-invalid-size-inputs-valid-test 
        --log-format json
        --log-interval 100 
        --patience 5
        --find-unused-parameters
        --save-dir  $$AMLT_OUTPUT_DIR/
```

We release our trained models in this link.

# Model inference
Please use the following script to retrieve passages for queries in NQ.
```bash
    - TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format dpr_qas --topics data/NQ/nq-test.qa.csv 
      --output_format dpr --output $$AMLT_OUTPUT_DIR/output_test.json 
      --checkpoint $$AMLT_OUTPUT_DIR/checkpoint_best.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 20 
      --beam 15
      --decode_query stable
      --fm_index data/fm_index/stable2/psgs_w100.fm_index 
```
# Evaluation
```bash
    - python3 seal/evaluate_output.py
      --file $$AMLT_OUTPUT_DIR/output_test.json 
```
# Contact
If there is any problem, please email liyongqi0@gmail.com. Please do not hesitate to email me directly as I do not frequently check GitHub issues.
