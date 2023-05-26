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
1) `data/NQ` folder. Please download `biencoder-nq-dev.json, biencoder-nq-train.json, nq-dev.qa.csv, nq-test.qa.csv` files into the `NQ` folder from the [DPR repositiory](https://github.com/facebookresearch/DPR).
2) `data/Trivia` folder. Please download `biencoder-trivia-dev.json, biencoder-trivia-train.json, trivia-dev.qa.csv, trivia-test.qa.csv` files into the `Trivia` folder from the [DPR repositiory](https://github.com/facebookresearch/DPR).
3) `data/fm_index/` folder. Please download fm_index files `psgs_w100.fm_index.fmi, psgs_w100.fm_index.oth` for the Wikipedia corpus and  `msmarco-passage-corpus.fm_index.fmi, msmarco-passage-corpus.fm_index.oth` for the MSMARCO corpus from this [link](https://drive.google.com/drive/folders/1xFs9rF49v3A-HODGp6W7mQ4hlo5FspnV?usp=sharing).
4) `data/training_data/` folder. 
   Download the `NQ_title_body_query_generated` from this [link](https://drive.google.com/drive/folders/1luVt0hNzAiRmWhtEmDy-apqC0ae2mn5V?usp=sharing).
   Download the `Trivia_title_body_query_generated` from this [link](https://drive.google.com/drive/folders/1rZ1ayY9Cx-gDfmTBImBpliTPJ4Ij1Qdk?usp=sharing).
   Download the `MSMARCO_title_body_query3` from this [link](https://drive.google.com/drive/folders/1bbqO7HII9_Ey7uOSi5NoPOAigs-55ov9?usp=sharing).

# Model training
We use the fairseq to train the BART_large model with the translation task. 
The script of training on NQ dataset is 
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
        
        
- name: GGR_ours_trivia_unsupervised
  sku: G8
  priority: High
  command:
    - export PATH=$$HOME/.local/bin/:$$PATH
    - fairseq-train
        $$AMLT_DATA_DIR/SEAL/Trivia_title_body_query_generated/bin 
        --finetune-from-model $$AMLT_DATA_DIR/SEAL/BART_FILES/bart.large/model.pt 
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
        --save-interval-updates 6000 
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
    - TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format dpr_qas --topics $$AMLT_DATA_DIR/Trivia/trivia-test.qa.csv
      --output_format dpr --output $$AMLT_OUTPUT_DIR/output_test.json 
      --checkpoint $$AMLT_OUTPUT_DIR/checkpoint_best.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 40 
      --beam 15
      --decode_query stable
      --fm_index $$AMLT_DATA_DIR/SEAL/fm_index/stable2/psgs_w100.fm_index
    - python3 seal/evaluate_output.py
      --file $$AMLT_OUTPUT_DIR/output_test.json

- name: GGR_ours_msmarco2
  sku: G8
  priority: High
  command:
    - export PATH=$$HOME/.local/bin/:$$PATH
    - fairseq-train
        $$AMLT_DATA_DIR/SEAL/MSMARCO_title_body_query3/bin 
        --finetune-from-model $$AMLT_DATA_DIR/SEAL/BART_FILES/bart.large/model.pt 
        --arch bart_large 
        --task translation 
        --criterion label_smoothed_cross_entropy 
        --source-lang source 
        --target-lang target 
        --truncate-source 
        --label-smoothing 0.1 
        --max-tokens 4096 
        --update-freq 1 
        --max-update 100000 
        --required-batch-size-multiple 1
        --validate-interval 1000000
        --save-interval 1000000
        --save-interval-updates 6000 
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
        --total-num-update 100000 
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
        --patience 3
        --find-unused-parameters
        --save-dir  $$AMLT_OUTPUT_DIR/
    - TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format msmarco --topics Tevatron/msmarco-passage
      --output_format msmarco --output $$AMLT_OUTPUT_DIR/output_test.json 
      --checkpoint $$AMLT_DATA_DIR/MSMARCO/checkpoint_best.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 10 
      --beam 7
      --decode_query stable
      --fm_index $$AMLT_DATA_DIR/SEAL/fm_index/stable2/msmarco-passage-corpus.fm_index
    - python3 seal/evaluate_output_msmarco.py
      $$AMLT_DATA_DIR/MSMARCO/qrels.msmarco-passage.dev-subset.txt $$AMLT_OUTPUT_DIR/output_test.json
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
