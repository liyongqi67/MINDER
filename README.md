# Introduction
We have published several works on generative retrieval as follows.
```
Multiview Identifiers Enhanced Generative Retrieval. ACL 2023. (MINDER)
Generative Retrieval for Conversational Question Answering. IPM 2023. (GCoQA)
Learning to Rank in Generative Retrieval. AAAI 2024. (LTRGR)
```
All code, data, and checkpoints of the above works are open-released. Please refer to the corresponding sections if you are interested: [MINDER](#MINDER), [GCoQA](https://github.com/liyongqi67/GCoQA), and [LTRGR](#LTRGR).
# MINDER
This is the official implementation for the paper "Multiview Identifiers Enhanced Generative Retrieval".  
The preprint version is released in [Arxiv](https://arxiv.org/abs/2305.16675).  
If you find our paper or code helpful,please consider citing as follows:
```bibtex
@inproceedings{li-etal-2023-multiview,
    title = "Multiview Identifiers Enhanced Generative Retrieval",
    author = "Li, Yongqi  and Yang, Nan  and Wang, Liang  and Wei, Furu  and Li, Wenjie",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    pages = "6636--6648",
}
```

## Install
```commandline
git clone https://github.com/liyongqi67/MINDER.git
sudo apt install swig
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -r requirements.txt
pip install -e .
```
## Data
Option 1: Please download the processed data into the `data` folder.
1) `data/NQ` folder. Please download `biencoder-nq-dev.json, biencoder-nq-train.json, nq-dev.qa.csv, nq-test.qa.csv` files into the `NQ` folder from the [DPR repositiory](https://github.com/facebookresearch/DPR).
2) `data/Trivia` folder. Please download `biencoder-trivia-dev.json, biencoder-trivia-train.json, trivia-dev.qa.csv, trivia-test.qa.csv` files into the `Trivia` folder from the [DPR repositiory](https://github.com/facebookresearch/DPR).
3) `data/MSMARCO` folder. Please download `qrels.msmarco-passage.dev-subset.txt` from this [link](https://drive.google.com/file/d/10P26nG02rwsLne2NJsooZVXvobPw9RLP/view?usp=sharing).
4) `data/fm_index/` folder. Please download fm_index files `psgs_w100.fm_index.fmi, psgs_w100.fm_index.oth` for the Wikipedia corpus and  `msmarco-passage-corpus.fm_index.fmi, msmarco-passage-corpus.fm_index.oth` for the MSMARCO corpus from this [link](https://drive.google.com/drive/folders/1xFs9rF49v3A-HODGp6W7mQ4hlo5FspnV?usp=sharing).
5) `data/training_data/` folder.   
   Download the `NQ_title_body_query_generated` from this [link](https://drive.google.com/drive/folders/1luVt0hNzAiRmWhtEmDy-apqC0ae2mn5V?usp=sharing).  
   Download the `Trivia_title_body_query_generated` from this [link](https://drive.google.com/drive/folders/1rZ1ayY9Cx-gDfmTBImBpliTPJ4Ij1Qdk?usp=sharing).  
   Download the `MSMARCO_title_body_query3` from this [link](https://drive.google.com/drive/folders/1bbqO7HII9_Ey7uOSi5NoPOAigs-55ov9?usp=sharing).
6) `data/pseudo_queries/` folder. [link](https://drive.google.com/drive/folders/10OIHLd5h81_qQ_TAPiU2gLVt3wsSWdS4?usp=drive_link)
   pseudo_queries for Wikipedia and MSMARCO.
  
Option2: You could process the data by yourself following the [instructions](https://github.com/liyongqi67/MINDER/blob/main/scripts/training/README.md).
## Model training
We use the fairseq to train the BART_large model with the translation task.  
The script for training on the NQ dataset is 
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
        --save-dir  ./
```
The script for training on the TriviaQA dataset is  
```bash
    - fairseq-train
        data/training_data/Trivia_title_body_query_generated/bin 
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
        --save-dir  ./
```
The script for training on the MSMARCO dataset is 
```bash
    - fairseq-train
        data/training_data/MSMARCO_title_body_query3/bin 
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
        --save-dir  ./
```
We trained the models on 8*32GB NVIDIA V100 GPUs. It took about 4d3h24m39s, 1d18h30m47s, 12h53m50s for training on NQ, TriviaQA, and MSMARCO, respectively.  
We release our trained model checkpoints in this [link](https://drive.google.com/drive/folders/1_EMelqpyJXhGcyCp9WjV1JZwGWxnZjQw?usp=sharing).

## Model inference
Please use the following script to retrieve passages for queries in NQ.
```bash
    - TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format dpr_qas --topics data/NQ/nq-test.qa.csv 
      --output_format dpr --output output_test.json 
      --checkpoint checkpoint_NQ.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 20 
      --beam 15
      --decode_query stable
      --fm_index data/fm_index/stable2/psgs_w100.fm_index 
```
Please use the following script to retrieve passages for queries in TriviaQA.
```bash
    - TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format dpr_qas --topics data/Trivia/trivia-test.qa.csv
      --output_format dpr --output output_test.json 
      --checkpoint checkpoint_TriviaQA.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 40 
      --beam 15
      --decode_query stable
      --fm_index data/fm_index/stable2/psgs_w100.fm_index
```
Please use the following script to retrieve passages for queries in MSMARCO.
```bash
    - TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format msmarco --topics Tevatron/msmarco-passage
      --output_format msmarco --output output_test.json 
      --checkpoint checkpoint_MSMARCO.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 10 
      --beam 7
      --decode_query stable
      --fm_index data/fm_index/stable2/msmarco-passage-corpus.fm_index
 ```
## Evaluation
Please use the following script to evaluate on NQ and TriviaQA.
```bash
    - python3 seal/evaluate_output.py
      --file output_test.json 
```
Please use the following script to evaluate on MSMARCO.
```bash
    - python3 seal/evaluate_output_msmarco.py
      data/MSMARCO/qrels.msmarco-passage.dev-subset.txt output_test.json
```
# LTRGR
This is the official implementation for the paper "Learning to Rank in Generative Retrieval".  
The preprint version is released in [Arxiv](https://arxiv.org/abs/2306.15222).
If you find our paper or code helpful,please consider citing as follows:
```bibtex
@article{li2023learning,
  title={Learning to rank in generative retrieval},
  author={Li, Yongqi and Yang, Nan and Wang, Liang and Wei, Furu and Li, Wenjie},
  journal={arXiv preprint arXiv:2306.15222},
  year={2023}
}
```

## Install
```commandline
git clone https://github.com/liyongqi67/MINDER.git
sudo apt install swig
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -r requirements.txt
pip install -e .
```
## Model Training
You could directly download our trained [checkpoints](https://drive.google.com/drive/folders/15Hwk_b1739nj9aICn6U1jQ1KPu6eAUXv?usp=sharing).
### Learning to generate
Learning to generate means to train the MINDER. You could refer to the above MINDER training procedures or load the trained [MINDER checkpoints](https://drive.google.com/drive/folders/1_EMelqpyJXhGcyCp9WjV1JZwGWxnZjQw?usp=sharing).  
### Learning to rank
Step 1: Data preparation. Load the MINDER checkpoints and obtain the top-200 retrieval results on the training set.  
On NQ
```bash
    TOKENIZERS_PARALLELISM=false python seal/search.py 
      --topics_format dpr_qas_train --topics data/NQ/biencoder-nq-train.json 
      --output_format dpr --output MINDER_NQ_train_top200.json 
      --checkpoint checkpoint_NQ.pt 
      --jobs 10 --progress --device cuda:0 --batch_size 10 
      --beam 15
      --decode_query stable
      --fm_index data/fm_index/stable2/psgs_w100.fm_index
      --include_keys
      --hits 200
```
Step 2: Train the model via rank loss.
```bash
    TOKENIZERS_PARALLELISM=false  python3 MINDER_learn_to_rank_v1.4.py 
        --checkpoint checkpoint_NQ.pt
        --fm_index data/fm_index/stable2/psgs_w100.fm_index
        --train_file MINDER_NQ_train_top200.json
        --do_fm_index True
        --per_gpu_train_batch_size 8
        --output_dir ./release_test
        --rescore_batch_size 70
        --num_train_epochs 3
        --factor_of_generation_loss 1000
        --rank_margin 300
        --shuffle_positives True
        --shuffle_negatives True
        --shuffle_negatives True
        --decode_query stable
        --pid2query pid2query_Wikipedia.pkl
```
## Model inference
Please use the following script to retrieve passages for queries in NQ.
```bash
for file in ./release_test/*
do
    if test -f $file
    then
        echo $file
        TOKENIZERS_PARALLELISM=false python -m seal.search \
        --topics_format dpr_qas --topics data/NQ/nq-test.qa.csv \
        --output_format dpr --output output_test.json \
        --checkpoint $file\
        --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
        --decode_query stable --fm_index data/fm_index/stable2/psgs_w100.fm_index --dont_fairseq_checkpoint
        python3 evaluate_output.py --file output_test.json
    fi
done

```
## Acknowledgments
Part of the code is based on [SEAL](https://github.com/facebookresearch/SEAL) and [sdsl-lite](https://github.com/simongog/sdsl-lite).
## Contact
If there is any problem, please email liyongqi0@gmail.com. Please do not hesitate to email me directly as I do not frequently check GitHub issues.
