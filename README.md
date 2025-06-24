# Introduction
We have published several works on generative retrieval as follows.
```
Multiview Identifiers Enhanced Generative Retrieval. ACL 2023. (MINDER)
Generative Retrieval for Conversational Question Answering. IPM 2023. (GCoQA)
Learning to Rank in Generative Retrieval. AAAI 2024. (LTRGR)
Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond. ACL 2024 (GRACE).
Distillation Enhanced Generative Retrieval. ACL 2024 findings (DGR).
```
All code, data, and checkpoints of the above works are open-released:  
1. MINDER, LTRGR, and DGR, are a series of works on text retrieval. LTRGR and DGR are continuously training based on the MINDER model, so we release MINDER, LTRGR, and DGR together in the same repository https://github.com/liyongqi67/MINDER.  
2. GCoQA is the work on conversational retrieval and is released at https://github.com/liyongqi67/GCoQA.  
3. GRACE is the work on cross-modal retrieval and is released at https://github.com/liyongqi67/GRACE.  
  
You could also refer to our preprint works on generative retrieval.
```
A Survey of Generative Search and Recommendation in the Era of Large Language Models.
Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation.
```
# MINDER
This is the official implementation for the paper "Multiview Identifiers Enhanced Generative Retrieval".  
The preprint version is released in [Arxiv](https://arxiv.org/abs/2305.16675).  
If you find our paper or code helpful, please consider citing as follows:
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
4) `data/fm_index/stable2/` folder. Please download fm_index files `psgs_w100.fm_index.fmi, psgs_w100.fm_index.oth` for the Wikipedia corpus and  `msmarco-passage-corpus.fm_index.fmi, msmarco-passage-corpus.fm_index.oth` for the MSMARCO corpus from this [link](https://drive.google.com/drive/folders/1xFs9rF49v3A-HODGp6W7mQ4hlo5FspnV?usp=sharing).
5) `data/training_data/` folder.   
   Download the `NQ_title_body_query_generated` from this [link](https://drive.google.com/drive/folders/1luVt0hNzAiRmWhtEmDy-apqC0ae2mn5V?usp=sharing).  
   Download the `Trivia_title_body_query_generated` from this [link](https://drive.google.com/drive/folders/1rZ1ayY9Cx-gDfmTBImBpliTPJ4Ij1Qdk?usp=sharing).  
   Download the `MSMARCO_title_body_query3` from this [link](https://drive.google.com/drive/folders/1bbqO7HII9_Ey7uOSi5NoPOAigs-55ov9?usp=sharing).
6) `data/pseudo_queries/` folder. [link](https://drive.google.com/drive/folders/10OIHLd5h81_qQ_TAPiU2gLVt3wsSWdS4?usp=drive_link)
   pseudo_queries for Wikipedia and MSMARCO.
  
Option 2: You could process the data by yourself following the [instructions](https://github.com/liyongqi67/MINDER/blob/main/scripts/training/README.md).
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
On MSMARCO  
```bash
     TOKENIZERS_PARALLELISM=false python seal/search.py 
       --topics_format msmarco_train --topics Tevatron/msmarco-passage
       --output_format dpr --output MINDER_MSMARCO_train_top100.json
       --checkpoint checkpoint_MSMARCO.pt 
       --jobs 5 --progress --device cuda:0 --batch_size 10 
       --beam 7
       --decode_query stable
       --fm_index data/fm_index/stable2/msmarco-passage-corpus.fm_index
       --include_keys --hits 100
 ```
Use the data_process3.py to transform the above "MINDER_NQ_train_top200.json" and "MINDER_MSMARCO_train_top100.json" files. (Please set the correct file path for the filename/output_file variables in data_process3.py.)

Step 2: Train the model via rank loss.  
On NQ
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
On MAMARCO  
```bash
    TOKENIZERS_PARALLELISM=false  python3 MINDER_learn_to_rank_v1.4.py 
        --checkpoint checkpoint_MSMARCO.pt
        --fm_index data/fm_index/stable2/msmarco-passage-corpus.fm_index
        --train_file MINDER_MSMARCO_train_top100.json
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
        --pid2query pid2query_msmarco.pkl
```

## Model inference
Please use the following script to retrieve passages.  
On NQ
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
On MSMARCO
```bash
for file in ./release_test/*
do
    if test -f $file
    then
       TOKENIZERS_PARALLELISM=false python seal/search.py 
       --topics_format msmarco --topics Tevatron/msmarco-passage
       --output_format msmarco --output output_test.json 
       --checkpoint $file 
       --jobs 5 --progress --device cuda:0 --batch_size 10 
       --beam 7
       --decode_query stable
       --fm_index data/fm_index/stable2/msmarco-passage-corpus.fm_index
       --dont_fairseq_checkpoint
      python3 evaluate_output_msmarco.py data/MSMARCO/qrels.msmarco-passage.dev-subset.txt output_test.json
    fi
done

```
# DGR
This is the official implementation for the paper "Distillation Enhanced Generative Retrieval".
The preprint version is released in [Arxiv](https://arxiv.org/pdf/2402.10769).
If you find our paper or code helpful,please consider citing as follows:
```bibtex
@misc{li2024distillation,
      title={Distillation Enhanced Generative Retrieval}, 
      author={Yongqi Li and Zhen Zhang and Wenjie Wang and Liqiang Nie and Wenjie Li and Tat-Seng Chua},
      year={2024},
      eprint={2402.10769},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Install
```commandline
git clone https://github.com/liyongqi67/MINDER.git
sudo apt install swig
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -r requirements_dgr.txt
pip install -e .
```
## Model Training
You could directly download our trained [DGR checkpoints](https://drive.google.com/drive/folders/1k-1cYzYl3GWeCI6f6ckUt8nCF3GmUZCo)
### Learning to generate
Learning to generate means to train the MINDER. You could refer to the above MINDER training procedures or load the trained [MINDER checkpoints](https://drive.google.com/drive/folders/1_EMelqpyJXhGcyCp9WjV1JZwGWxnZjQw?usp=sharing).  
### Distillation enhanced generative retrieval training

#### Get training data
Step 1: Data preparation. Load the MINDER checkpoints and obtain the top-200 retrieval results on the training set.  
Take NQ as an example. For TriviaQA and MsMarco, you may change data and checkpoints, which can be download from MINDER.
```bash
TOKENIZERS_PARALLELISM=false python seal/search.py 
    --topics_format dpr_qas_train --topics data/NQ/biencoder-nq-train.json 
    --output_format dpr --output ./TrainData/NQ/MINDER_NQ_train_top200.json 
    --checkpoint checkpoint_NQ.pt 
    --jobs 10 --progress --device cuda:0 --batch_size 10 
    --beam 15
    --decode_query stable
    --fm_index data/fm_index/stable2/psgs_w100.fm_index
    --include_keys
    --hits 200
```

Step 2: Obtain ranking score of teacher model for the training question-answer pairs. We use E5 and SimLM as ranking teacher.
On NQ and SimLM as teacher(use simLM specially trained on NQ data), which can be download  [DGR checkpoints](https://drive.google.com/drive/folders/1k-1cYzYl3GWeCI6f6ckUt8nCF3GmUZCo).
```bash
python ./seal/check_top200.py  
--teacher_output_path ./TrainData/simLM_NQ 
--original_path ./TrainData/NQ  
--do_train Ture --teacher_path ./nq-simLM-reranker/reranker-model
--teacher_name simLM --teacher_kind score_teacher_simLM_NQ 
--original_file MINDER_NQ_train_top200.json
```
On NQ and E5 as teacher
```bash
python ./seal/check_top200.py  
--teacher_output_path ./TrainData/E5_NQ 
--original_path ./TrainData/NQ
--do_train Ture --teacher_path intfloat/e5-large-v2
--teacher_name E5 --teacher_kind score_teacher_E5
--original_file MINDER_NQ_train_top200.json
```

On TriviaQA and SimLM as teacher(use simLM specially trained on NQ data)
```bash
python ./seal/check_top200.py 
--teacher_output_path ./TrainData/simLM_Trival_QA  
--original_path ./Train_data_Triva_QA --teacher_path ./nq-simLM-reranker/reranker-model
--teacher_name simLM --teacher_kind score_teacher_simLM_NQ 
--original_file MINDER_triviaQA_train_top200.json
```

On TriviaQA and E5 as teacher
```bash
python ./seal/check_top200.py 
--teacher_output_path ./TrainData/simLM_Trival_QA  
--do_train Ture --teacher_path intfloat/e5-large-v2
--teacher_name E5 --teacher_kind score_teacher_E5
--original_file MINDER_triviaQA_train_top200.json
```

On MsMarco and simLM as teacher
```bash
python ./seal/check_top200.py  
--teacher_output_path ./TrainData/simLM_MSMarco 
--original_path ./TrainData/MSMarco --do_train Ture --teacher_path intfloat/simlm-msmarco-reranker 
--teacher_name simLM_msmarco --teacher_kind score_teacher_simLM_msmarco 
--if_MSMarco True
```
####  Distillation enhanced generative retrieval training process
train the model via distilled RankNet loss.
On NQ and simLM as teacher
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 ./seal/minder_ltr_listwise.py --decode_query stable --train_file ./TrainData/simLM_NQ/ --per_gpu_train_batch_size 1  --factor_of_generation_loss 500 --shuffle_positives True --shuffle_negatives True --do_fm_index True --sort_negative_only True --order_margin True --rest_all_negative False --sample_length 6 --num_train_epochs 3 --rank_margin 300 --teacher_kind score_teacher_simLM_NQ --manual_rankNet True
```
For training on NQ based on E5 as teacher, we only need to change 'teacher_kind' to 'score_teacher_E5'
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 ./seal/minder_ltr_listwise.py --decode_query stable --train_file ./TrainData/E5_NQ/ --per_gpu_train_batch_size 1  --factor_of_generation_loss 500 --shuffle_positives True --shuffle_negatives True --do_fm_index True --sort_negative_only True --order_margin True --rest_all_negative False --sample_length 6 --num_train_epochs 3 --rank_margin 300 --teacher_kind score_teacher_E5 --manual_rankNet True
```

For TriviaQA, we only need to change training data 'train_file' to './TrainData/simLM_Trival_QA/' and teacher_kind to 'score_teacher_simLM_NQ'

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 ./seal/minder_ltr_listwise.py --decode_query stable --train_file ./TrainData/simLM_Trival_QA/ --per_gpu_train_batch_size 1  --factor_of_generation_loss 500 --shuffle_positives True --shuffle_negatives True --do_fm_index True --sort_negative_only True --order_margin True --rest_all_negative False --sample_length 6 --num_train_epochs 3 --rank_margin 300 --teacher_kind score_teacher_simLM_NQ --manual_rankNet True
```
For MsMarco, we only need to change training data 'train_file' to './TrainData/simLM_MSMarco/', and change 'teacher_kind' to 'score_teacher_simLM_msmarco'


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 ./seal/minder_ltr_listwise.py --decode_query stable --train_file ./TrainData/MSMarco_train_all_simLM_NQ_marco/ --per_gpu_train_batch_size 1  --factor_of_generation_loss 500 --shuffle_positives True --shuffle_negatives True --do_fm_index True --sort_negative_only True --order_margin True --rest_all_negative False --sample_length 6 --num_train_epochs 3 --rank_margin 300 --teacher_kind score_teacher_simLM_msmarco --manual_rankNet True --checkpoint ./checkpoint_MSMARCO.pt  --fm_index ./data/fm_index/msmarco-passage-corpus.fm_index
--pid2query ./data/pid2query_msmarco.pkl
```

To train model with learning to rank loss such as listNet loss, ListMLE, LambdaLoss, you can clone the repo [allRank](https://github.com/allegro/allRank) into the root directory of this project. Then run
```bash
PWD = `pwd`
export PYTHONPATH=$PWD:$PWD/allRank:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --standalone --nnodes=1 --nproc-per-node=4 ./seal/minder_ltr_listwise.py --decode_query stable --train_file ./TrainData/simLM_NQ/  --per_gpu_train_batch_size 1 --experiment_name SimLM_NQ_lbdloss_weight3_list2_m300 --scenario_name   updated_two_posi_sort_nega_only_anlysis  --shuffle_positives True --shuffle_negatives True  --do_fm_index True --sort_negative_only True --order_margin True --rest_all_negative False --sample_length 6 --num_train_epochs 3 --teacher_kind score_teacher_simLM_NQ --factor_of_generation_loss 1 --margin_use_ltrlloss True --loss_func lambdaLoss
```

## Model Inference
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
