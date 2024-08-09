# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import os
import socket
from pathlib import Path

import argparse
import logging
import os
import random
import glob
import timeit
import json
from seal import keys as rk

from tqdm import tqdm, trange
from copy import copy
import re
import torch
torch.set_printoptions(profile="full")
import copy
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import wandb
import transformers
from transformers import T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup



import pickle
from torch.cuda.amp import autocast as autocast
import numpy as np
from datasets import load_dataset
from multiprocessing import Pool


from contextlib import contextmanager


from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizer, BartForConditionalGeneration
from seal.retrieval import SEALSearcher
from torch.utils.data import Dataset
from seal import SEALSearcher
from random import choice
import random
from loss_teacher import listNet, listNet_add_norm, KLloss,listNet_org, KLloss_add_Norm_fixed,rankNet,lambdaLoss, lambdaLoss_add_norm,rankNet_add_norm
from loss_teacher import ndcgLoss2_scheme, ndcgLoss1_scheme
from allRank.allrank.models import losses

# In[2]:
np_str_obj_array_pattern = re.compile(r'[SaUO]')

logger = logging.getLogger(__name__)


transformers.logging.set_verbosity_error()
import json
from tqdm import tqdm
import unicodedata

import json, string, re
from collections import Counter, defaultdict
from argparse import ArgumentParser
import unicodedata
import regex
import logging
import copy
import math
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import ftfy
from fuzzywuzzy import fuzz
class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups
class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)
def _normalize(text):
    return unicodedata.normalize("NFD", text)
def has_answer(answers, text, match_type="string") -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return 1.0

    # elif match_type == "regex":
    #     # Answer is a regex
    #     for single_answer in answers:
    #         single_answer = _normalize(single_answer)
    #         if regex_match(text, single_answer):
    #             return True
    return 0.0
# In[3]:

banned = set(stopwords.words('english'))

def span_iterator(tokens, ngrams=3, banned=banned):
    
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i+ngrams)
def extract_spans(text, source, n_samples, min_length, max_length, temperature=1.0):
    source = source.split("||", 1)[0]
    query_tokens = source.split()
    query_tokens_lower = [t.lower() for t in query_tokens]

    passage_tokens = text.split()
    passage_tokens_lower = [t.lower() for t in passage_tokens]

    matches = defaultdict(int)

    for i1, _ in enumerate(query_tokens_lower):
        j1 = i1+3
        str_1 = " ".join(query_tokens_lower[i1:j1])

        for (i2, j2) in span_iterator(passage_tokens_lower, 3):
            str_2 = " ".join(passage_tokens_lower[i2:j2])
            ratio = fuzz.ratio(str_1, str_2) / 100.0
            matches[i2] += ratio

    if not matches:
        indices = [0]

    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -(x[1])))
        weights = list(weights)
        sum_weights = float(sum([0] + weights))
        if sum_weights == 0.0 or not weights:
            indices = [0]
            weights = [1.0]
        else:
            weights = [math.exp(float(w) / temperature) for w in weights]
            Z = sum(weights)
            weights = [w / Z for w in weights]

        indices = random.choices(indices, weights=weights, k=n_samples)

    span_list=[]
    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i+subspan_size])
        span_list.append(span)
    return span_list

class FinetuningDataset(Dataset):
    def __init__(self, filename, tokenizer,load_small, top_p, top_ngram, margin, pid2query, 
                  shuffle_positives,shuffle_negatives,sample_length, use_top10,distill_list_num,teacher_kind = "score_teacher_E5", rest_all_negative= True,LTR_without_teacher = False ):
        

        self._load_small = load_small
        
        # for args.LTR_without_teacher
        self.shuffle_positives = shuffle_positives
        self.shuffle_negatives = shuffle_negatives
        self.top_p = top_p
        self.top_ngram = top_ngram
        self.tokenizer = tokenizer
        self.margin = margin
        self.teacher_kind = teacher_kind
        self.rest_all_negative = rest_all_negative
        self.sample_length = sample_length
        self.LTR_without_teacher = LTR_without_teacher
        self.use_top10 = use_top10
        self.distill_list_num = distill_list_num
        

        self.data = []
        if pid2query:
            with open(pid2query, 'rb') as f:
                self.pid2query = pickle.load(f)
        
        if os.path.isfile(filename):
            with open(filename, 'r') as load_f:
                for line in tqdm(load_f):
                    line = json.loads(line)
                    if self.use_top10:
                            self.data.append(line)
                    else:   
                        if len(line['positive_ctxs'])>0  and len(line['negative_ctxs']) > 15:                        
                            if line['positive_ctxs'][0]['score'] < line['negative_ctxs'][0]['score']+self.margin :                            
                                new_dict = {}
                                new_dict["question"] = line["question"]
                                new_dict["answers"] = line["answers"]
                                positive_ctxs = []
                                negative_ctxs = []                                
                                        
                                positive_ctxs.append(line["positive_ctxs"][0])

                                if len(line['positive_ctxs'])>5:
                                    #positive_ctxs.append(line["positive_ctxs"][1])
                                    rand_positive = random.sample(line["positive_ctxs"][1:],  4)
                                    positive_ctxs.extend([posi for posi in rand_positive])
                                if len(line['positive_ctxs'])> 1 and len(line['positive_ctxs']) <= 5:
                                    positive_ctxs.extend([line["positive_ctxs"][posi_idx] for posi_idx in range(len(line["positive_ctxs"])) if posi_idx != 0])
                                negative_ctxs.append(line["negative_ctxs"][0])
                                ## top negative
                                # negative_ctxs.append(line["negative_ctxs"][1])

                                rand = random.sample(line["negative_ctxs"][1:],  14)
                                negative_ctxs.extend([nega for nega in rand])
                                # rand = random.sample(line["negative_ctxs"],  15)
                                # negative_ctxs.extend([nega for nega in rand])
                                
                                # # for Msmarco
                                # negative_ctxs.extend([nega for nega in line["negative_ctxs"]])

                                # ## top negative
                                # rand =   line["negative_ctxs"][:15]
                                # negative_ctxs.extend([nega for nega in rand])
                                
                                new_dict['positive_ctxs'] = positive_ctxs
                                new_dict['negative_ctxs'] = negative_ctxs
                                self.data.append(new_dict)    

        else:
            for file in os.listdir(filename):
                with open(filename+file, 'r') as load_f:
                    for line in tqdm(load_f):
                        line = json.loads(line)
                        if self.use_top10:
                                self.data.append(line)
                        else: 
                            if len(line['positive_ctxs'])>0 and len(line['negative_ctxs'])> 15:                                                 
                                if line['positive_ctxs'][0]['score'] < line['negative_ctxs'][0]['score']+self.margin:                                
                                    new_dict = {}
                                    new_dict["question"] = line["question"]
                                    new_dict["answers"] = line["answers"]
                                    positive_ctxs = []
                                    negative_ctxs = []                                                                            
                                    positive_ctxs.append(line["positive_ctxs"][0])                                    
                                    
                                    if len(line['positive_ctxs'])>5:

                                        #positive_ctxs.append(line["positive_ctxs"][1])
                                        rand_positive = random.sample(line["positive_ctxs"][1:],  4)
                                        positive_ctxs.extend([posi for posi in rand_positive])
                                    if len(line['positive_ctxs'])> 1 and len(line['positive_ctxs']) <= 5:
                                        positive_ctxs.extend([line["positive_ctxs"][posi_idx] for posi_idx in range(len(line["positive_ctxs"])) if posi_idx != 0])
                                    negative_ctxs.append(line["negative_ctxs"][0])
                                    
                                    # ## top negative
                                    # negative_ctxs.append(line["negative_ctxs"][1])
                                    

                                    rand = random.sample(line["negative_ctxs"][1:],  14)
                                    negative_ctxs.extend([nega for nega in rand])
                                    # rand = random.sample(line["negative_ctxs"],  15)
                                    # negative_ctxs.extend([nega for nega in rand])

                                    ## top negative
                                    # rand =   line["negative_ctxs"][:15]
                                    # negative_ctxs.extend([nega for nega in rand])

                                    # # for Msmarco
                                    # negative_ctxs.extend([nega for nega in line["negative_ctxs"]])
                                                        
                                    new_dict['positive_ctxs'] = positive_ctxs
                                    new_dict['negative_ctxs'] = negative_ctxs
                                    self.data.append(new_dict)  

                                                        
        self._total_data = 0
        if self._load_small:
            self._total_data = 100
        else:
            self._total_data = len(self.data)

    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
  
        entry = self.data[idx]
        # print("entry:",entry)
        passages = []
        teacher_score = []
        minder_score = []

        if self.use_top10:
            #print("------use_top10")
            teacher_score_whole = []
            passages_whole = []
            positive_ctxs = [entry['positive_ctxs']]
            passages_whole = [pas for pas in entry['train_ctxs'] ]
            passages = passages_whole[ : self.sample_length]
            teacher_score_whole = [pas[self.teacher_kind] for pas in entry['train_ctxs']]
            teacher_score = teacher_score_whole[ : self.sample_length]
            # teacher_score = torch.tensor(teacher_score)
            # print("teacher_score:", teacher_score)
            minder_score.extend([pas['score'] for pas in passages]) 

            teacher_score = torch.tensor(teacher_score )
            minder_score = torch.tensor(minder_score )
            
            # for rankNet 
            # if args.use_top10:
            #     #print("teacher_score_before-----", teacher_score)
            #     _, sorted_indices = torch.sort(teacher_score, descending=True)
            #    # 获取每个原始索引在排序后的新位置
            #     desired_order_indices = torch.argsort(sorted_indices)
            #     print("desired_order_indices", desired_order_indices)
            #     desired_order_indices.add_(1) 
            #     teacher_score = 1.0 / desired_order_indices.float()
            #     #print("teacher_score_after-----", teacher_score)

            
            if self.teacher_kind == "score_teacher_E5":
                #print("with temperature")
                teacher_score /= 0.01
            
                        
        else:
            positive_ctxs = entry['positive_ctxs']
            negative_ctxs = entry['negative_ctxs']

            if self.LTR_without_teacher:
                #print("-----LTR_without_teacher------")
                passages = []
                passages.append(positive_ctxs[0])
                passages.append(negative_ctxs[0])
                

                if self.shuffle_positives== True:
                    passages.append(choice(positive_ctxs))
                else:
                    passages.append(positive_ctxs[0])

                if self.shuffle_negatives == True:
                    passages.append(choice(negative_ctxs))
                else:
                    passages.append(negative_ctxs[0])
                minder_score.extend([pas['score'] for pas in passages])
            else: 

                # zz 组装teacher 和 minder score
                passages.append(positive_ctxs[0])
                teacher_score.append(positive_ctxs[0][self.teacher_kind])
                
                
                if not self.rest_all_negative:
                    # passages.append(positive_ctxs[1])
                    # teacher_score.append(positive_ctxs[1][self.teacher_kind])
                    #print("----choose 2 positive----")
                    posi_sample = random.choice(positive_ctxs)                
                    passages.append(posi_sample)
                    teacher_score.append(posi_sample[self.teacher_kind])
                    if self.distill_list_num == 4:
                    # choose 4 positive
                        posi_sample = random.choice(positive_ctxs)                
                        passages.append(posi_sample)
                        teacher_score.append(posi_sample[self.teacher_kind])

                        posi_sample = random.choice(positive_ctxs)                
                        passages.append(posi_sample)
                        teacher_score.append(posi_sample[self.teacher_kind])

                    
                passages.append(negative_ctxs[0])                
                teacher_score.append(negative_ctxs[0][self.teacher_kind])
                ## top negative
                # passages.append(negative_ctxs[1])                
                # teacher_score.append(negative_ctxs[1][self.teacher_kind])
                
                #top4 negative for analysis experiment of sample stratedgy
                # passages.extend(negative_ctxs[0:4])
                # for ctx in negative_ctxs[0:4]:
                #     teacher_score.append(ctx[self.teacher_kind])
                
                len_passage = len(passages)
                if self.sample_length - len_passage >0:
                    #print("random_sample")
                    #print(self.sample_length, len_passage,)
                    nega_samples = random.sample(negative_ctxs[1:], self.sample_length - len_passage) #需要改到1
                    # print("nega_samples:", nega_samples)
                    passages.extend(nega_samples)


                    teacher_score.extend([nega[self.teacher_kind] for nega in nega_samples])
                
                minder_score.extend([pas['score'] for pas in passages])   
                    
                # temperature
                teacher_score = torch.tensor(teacher_score )
                minder_score = torch.tensor(minder_score )
                # print("teacher_score:", teacher_score)
                if self.teacher_kind == "score_teacher_E5":
                    print("with temperature")
                    teacher_score /= 0.01
                # print("teacher_score with temp:", teacher_score)
                # print("teacher_score_1:", teacher_score)

        title_target=" " +positive_ctxs[0]['title'].strip() + " @@"
        span_target=" " +extract_spans(positive_ctxs[0]['text'], entry['question']+" || body"+" || +", 1, 10, 10, temperature=1.0)[0]
        query_target = " " +choice(self.pid2query[positive_ctxs[0]["passage_id"].strip()])


        found_keys = []
        for s in passages:
            if len(s['keys']) >= self.top_ngram:
                for key in s['keys'][:self.top_ngram]:
                    found_keys.append([key[2], key[0], key[1]])
            else:
                for key in s['keys']:
                    input_ids = self.tokenizer(key[0], padding=False)['input_ids']
                    if input_ids[-1] in searcher.strip_token_ids:
                        input_ids = input_ids[:-1] 
                    if input_ids[0] in searcher.strip_token_ids:
                        input_ids = input_ids[1:] 
                    found_keys.append([key[2], key[0], key[1]])
                for j in range(self.top_ngram - len(s['keys'])):
                    input_ids = self.tokenizer('padding token key', padding=False)['input_ids']
                    if input_ids[-1] in searcher.strip_token_ids:
                        input_ids = input_ids[:-1] 
                    if input_ids[0] in searcher.strip_token_ids:
                        input_ids = input_ids[1:] 
                    #found_keys.append([0.0, key[0], 10000])
                    found_keys.append([0.0, 'pad token', 1000000]) #####12/25


        if self.LTR_without_teacher:            
            return_feature_dict = { 'question': entry['question'], 
                                    'found_keys': found_keys,
                                    'title_target':title_target,
                                    'span_target':span_target,
                                    'query_target':query_target,
                                    "minder_score": minder_score,
                                    }  
        else:
            return_feature_dict = { 'question': entry['question'], 
                                   'answer': '*'.join(entry['answers']),
                                    'found_keys': found_keys,
                                    'title_target':title_target,
                                    'span_target':span_target,
                                    'query_target':query_target,
                                    "teacher_score": teacher_score,
                                    "minder_score": minder_score,
                                    }           
        return return_feature_dict

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#######################################################yongqi
def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def prefix_allowed_tokens_fn(batch_id, sent):
    return decoder_trie.get(sent.tolist())

def dist_gather_tensor(t):
    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[torch.distributed.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors

def repetition(ngram, coverage):
    if not coverage:
        return 1.0
    ngram = set(ngram)
    coeff = 1.0 - 0.8 + (0.8 * len(ngram.difference(coverage)) / len(ngram))
    return coeff
def model_forward(args,batch,model,ntokens):
            with torch.no_grad():
                question = batch['question']
                answer = batch['answer']
                found_keys=batch['found_keys']
                title_target=batch['title_target']
                span_target=batch['span_target']
                #query_target=batch['query_target']
                minder_score = batch['minder_score']
                if not args.LTR_without_teacher:
                    #print("----has teacher-----")
                    teacher_score = batch["teacher_score"]
                # print("batch: ",batch)

                question = [(" " + q.strip()) if searcher.prepend_space else q.strip() for q in question]
                new_found_keys = [[] for _ in range(len(question))]
                count = [[] for _ in range(len(question))]
                for kk in found_keys:
                    for j in range(len(question)):
                        input_ids = searcher.bart_tokenizer(kk[1][j], padding=False)['input_ids']
                        if input_ids[-1] in searcher.strip_token_ids:
                            input_ids = input_ids[:-1] 
                        if input_ids[0] in searcher.strip_token_ids:
                            input_ids = input_ids[1:] 
                        new_found_keys[j].append((kk[0][j].item(),input_ids))
                        count[j].append(kk[2][j].item())
                found_keys = new_found_keys


                ngrams_len = []
                title_index = []
                body_index = []
                query_index = []
                coeff = []
                for kk in found_keys:
                    current_coverage = set()
                    i=0
                    for s, k in kk:
                        if i%args.top_ngram == 0:
                            current_coverage = set()
                        if k[0] ==searcher.title_bos_token_id and k[-1] == searcher.title_eos_token_id:
                            title_index.append(1)
                            body_index.append(0)
                            #query_index.append(0)
                        elif k[0]== 45056:
                            title_index.append(0)
                            body_index.append(0)
                            #query_index.append(1)
                        else:
                            title_index.append(0)
                            body_index.append(1)
                            #query_index.append(0)  
                        ngrams_len.append(len(k))  
                        tts = set(k)
                        coeff.append(repetition(tts, current_coverage))
                        current_coverage |= tts 
                        i+=1      

            #model.eval()
            model.train()
            if searcher.decode_titles:
                if searcher.use_markers:
                    batch_str = [i + " || title" for i in question]
                if searcher.value_conditioning:
                    batch_str = [i + " || +" for i in batch_str]

                #####generation_loss_title#############
                encoding = searcher.bart_tokenizer(batch_str,padding="longest",truncation=True,max_length=50,return_tensors="pt")
                target_encoding = searcher.bart_tokenizer(title_target,padding="longest",truncation=True,max_length=15,return_tensors="pt")
                # labels = target_encoding.input_ids
                # labels[labels == searcher.bart_tokenizer.pad_token_id] = -100

                #############
                decoder_input_ids = target_encoding.input_ids[:, :-1].clone() # without eos/last pad
                labels = target_encoding.input_ids[:, 1:].clone().to(model.device) # without bos
                decoder_attention_mask = (decoder_input_ids != searcher.bart_tokenizer.pad_token_id)
                # breakpoint()
                outputs = model(
                        encoding.input_ids.to(model.device),
                        attention_mask=encoding.attention_mask.to(model.device),
                        decoder_input_ids=decoder_input_ids.to(model.device),
                        decoder_attention_mask=decoder_attention_mask.to(model.device),
                        use_cache=False
                        )

                lm_logits = outputs[0]
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=searcher.bart_tokenizer.pad_token_id)

                generation_loss_title = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
                ###########################
                #generation_loss_title=model(input_ids=encoding.input_ids.to(model.device), attention_mask=encoding.attention_mask.to(model.device), labels=labels.to(model.device)).loss
                #####generation_loss_title#############

                input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']
                if args.local_rank != -1:
                    scores = rk.rescore_keys_with_grad(
                            model.module,
                            input_tokens,
                            found_keys,
                            batch_size=args.rescore_batch_size,
                            length_penalty=0.0,
                            strip_from_bos=[
                                searcher.title_bos_token_id,
                                searcher.code_bos_token_id,
                                searcher.bart_model.config.decoder_start_token_id],
                            strip_from_eos=[searcher.bart_model.config.eos_token_id])
                else:
                    scores = rk.rescore_keys_with_grad(
                        model,
                        input_tokens,
                        found_keys,
                        batch_size=args.rescore_batch_size,
                        length_penalty=0.0,
                        strip_from_bos=[
                            searcher.title_bos_token_id,
                            searcher.code_bos_token_id,
                            searcher.bart_model.config.decoder_start_token_id],
                        strip_from_eos=[searcher.bart_model.config.eos_token_id])
                scores_title = scores.reshape(-1)
            if searcher.decode_body:
                if searcher.use_markers:
                    batch_str = [i + " || body" for i in question]
                if searcher.value_conditioning:
                    batch_str = [i + " || +" for i in batch_str]
                #####generation_loss_span#############
                encoding = searcher.bart_tokenizer(batch_str,padding="longest",truncation=True,max_length=50,return_tensors="pt")
                target_encoding = searcher.bart_tokenizer(span_target,padding="longest",truncation=True,max_length=15,return_tensors="pt")
                # labels = target_encoding.input_ids
                # labels[labels == searcher.bart_tokenizer.pad_token_id] = -100

                #############
                decoder_input_ids = target_encoding.input_ids[:, :-1].clone() # without eos/last pad
                labels = target_encoding.input_ids[:, 1:].clone().to(model.device) # without bos
                decoder_attention_mask = (decoder_input_ids != searcher.bart_tokenizer.pad_token_id)
                # breakpoint()
                outputs = model(
                        encoding.input_ids.to(model.device),
                        attention_mask=encoding.attention_mask.to(model.device),
                        decoder_input_ids=decoder_input_ids.to(model.device),
                        decoder_attention_mask=decoder_attention_mask.to(model.device),
                        use_cache=False
                        )

                lm_logits = outputs[0]
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=searcher.bart_tokenizer.pad_token_id)

                generation_loss_span = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
                ###########################

                # generation_loss_span=model(input_ids=encoding.input_ids.to(model.device), attention_mask=encoding.attention_mask.to(model.device), labels=labels.to(model.device)).loss
                #####generation_loss_span#############

                input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']
                if args.local_rank != -1:
                    scores = rk.rescore_keys_with_grad(
                            model.module,
                            input_tokens,
                            found_keys,
                            batch_size=args.rescore_batch_size,
                            length_penalty=0.0,
                            strip_from_bos=[
                                searcher.title_bos_token_id,
                                searcher.code_bos_token_id,
                                searcher.bart_model.config.decoder_start_token_id],
                            strip_from_eos=[
                                searcher.title_eos_token_id,
                                searcher.code_eos_token_id,
                                searcher.bart_model.config.eos_token_id])
                else:
                    scores = rk.rescore_keys_with_grad(
                        model,
                        input_tokens,
                        found_keys,
                        batch_size=args.rescore_batch_size,
                        length_penalty=0.0,
                        strip_from_bos=[
                            searcher.title_bos_token_id,
                            searcher.code_bos_token_id,
                            searcher.bart_model.config.decoder_start_token_id],
                        strip_from_eos=[
                            searcher.title_eos_token_id,
                            searcher.code_eos_token_id,
                            searcher.bart_model.config.eos_token_id])
                scores_body = scores.reshape(-1)
            '''
            if searcher.decode_query == "stable":
                if searcher.use_markers:
                    batch_str = [i + " || query" for i in batch_str]
                if searcher.value_conditioning:
                    batch_str = [i + " || +" for i in batch_str]

                #####generation_loss_title#############
                encoding = searcher.bart_tokenizer(batch_str,padding="longest",truncation=True,max_length=50,return_tensors="pt")
                target_encoding = searcher.bart_tokenizer(title_target,padding="longest",truncation=True,max_length=30,return_tensors="pt")
                # labels = target_encoding.input_ids
                # labels[labels == searcher.bart_tokenizer.pad_token_id] = -100

                #############
                decoder_input_ids = target_encoding.input_ids[:, :-1].clone() # without eos/last pad
                labels = target_encoding.input_ids[:, 1:].clone().to(model.device) # without bos
                decoder_attention_mask = (decoder_input_ids != searcher.bart_tokenizer.pad_token_id)
                # breakpoint()
                outputs = model(
                        encoding.input_ids.to(model.device),
                        attention_mask=encoding.attention_mask.to(model.device),
                        decoder_input_ids=decoder_input_ids.to(model.device),
                        decoder_attention_mask=decoder_attention_mask.to(model.device),
                        use_cache=False
                        )

                lm_logits = outputs[0]
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=searcher.bart_tokenizer.pad_token_id)

                generation_loss_query = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
                ###########################
                #generation_loss_title=model(input_ids=encoding.input_ids.to(model.device), attention_mask=encoding.attention_mask.to(model.device), labels=labels.to(model.device)).loss
                #####generation_loss_title#############

                input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']
                if args.local_rank != -1:
                    scores = rk.rescore_keys_with_grad(
                            model.module,
                            input_tokens,
                            found_keys,
                            batch_size=args.rescore_batch_size,
                            length_penalty=0.0,
                            strip_from_bos=[
                                searcher.title_bos_token_id,
                                searcher.query_bos_token_id,
                                searcher.code_bos_token_id,
                                searcher.bart_model.config.decoder_start_token_id],
                            strip_from_eos=[searcher.bart_model.config.eos_token_id])

                else:
                    scores = rk.rescore_keys_with_grad(
                        model,
                        input_tokens,
                        found_keys,
                        batch_size=args.rescore_batch_size,
                        length_penalty=0.0,
                        strip_from_bos=[
                            searcher.title_bos_token_id,
                            searcher.query_bos_token_id,
                            searcher.code_bos_token_id,
                            searcher.bart_model.config.decoder_start_token_id],
                        strip_from_eos=[searcher.bart_model.config.eos_token_id])
                scores_query = scores.reshape(-1)

                #####12/25
                scores_query = torch.minimum(scores_query + 14, torch.tensor(-1.0, dtype=torch.float32))
                #####12/25
            '''
            
            #print("title body query-----", scores_title, scores_body, scores_query)
            #scores = scores_title*torch.Tensor(title_index).to(scores_title.device) + scores_body*torch.Tensor(body_index).to(scores_title.device) +  scores_query*torch.Tensor(query_index).to(scores_query.device)
            scores = scores_title*torch.Tensor(title_index).to(scores_title.device) + scores_body*torch.Tensor(body_index).to(scores_title.device)

            ########################
            if args.do_fm_index:
                count = torch.Tensor(count).to(scores.device)
                count = count.reshape(-1)

                scores -= 1e-10

                snr = torch.log((count + searcher.smoothing) / (ntokens + searcher.smoothing))

                scores = \
                    (scores + torch.log(1 - torch.exp(snr)+1e-10)) - \
                    (snr + torch.log(1 - torch.exp(scores)+1e-10))

                scores = torch.where(scores < 0.0, torch.zeros_like(scores), scores)
                scores = scores**searcher.score_exponent

                coeff = torch.Tensor(coeff).to(scores.device).reshape(-1)

                scores = scores*coeff

            #########################
            scores = torch.reshape(scores,(-1,args.top_ngram))
            #print("scores1----", scores)

            scores = torch.sum(scores, dim=1)
            #print("sum---scores1----", scores)

            scores = torch.reshape(scores, (-1, args.sample_length))
            #print("reshape scores-----: ", scores)
            #要改的            
            
            if args.LTR_without_teacher:
                print("------reproduce LTR without teacher------")
                new_scores1 = scores[:,1] - scores[:,0] + args.rank_margin
                #print("new_scores1: ", new_scores1)
                # print('postive',scores[:,0])
                # print('negative',scores[:,1])
                rank_loss1 = torch.mean(torch.where(new_scores1 < 0.0, torch.zeros_like(new_scores1), new_scores1))

                new_scores2 = scores[:,3] - scores[:,2] + args.rank_margin + 100
                # print('postive',scores[:,0])
                # print('negative',scores[:,1])
                rank_loss2 = torch.mean(torch.where(new_scores2 < 0.0, torch.zeros_like(new_scores2), new_scores2))

                rank_loss = 0.5*rank_loss1 + 0.5*rank_loss2
                #rank_loss = 0.7*rank_loss1 + 0.3*rank_loss2
                #print("LTR_rank_loss:  ",rank_loss1, rank_loss2,rank_loss, sep = "-----")
                        
            else:
                teacher_score = teacher_score.to(args.device)
                print("------minder_score: ", minder_score)
                print("-----model score: ", scores)
                print("-----teacher score: ", teacher_score)
                # print(scores, teacher_score, sep = "------")
                
                # manual margin
                if args.order_margin:
                    #print("-----manual list order margin------- ")
                    
                    #use_two_positive, order negative passage by teacher
                    if args.sort_negative_only and not args.rest_all_negative:
                        #print("-----choose two positive ----")
                       
                        sorted_scores = []
                        margin_teacher = []
                        sorted_teachers = []
                        # #reproduce LTR
                        # if args.sample_length == 4:
                        #     new_scores1 = scores[:,2] - scores[:,0] + args.rank_margin
                                            
                        #     rank_loss1 = torch.mean(torch.where(new_scores1 < 0.0, torch.zeros_like(new_scores1), new_scores1))

                        #     new_scores2 = scores[:,3] - scores[:,1] + args.rank_margin + 100
                            
                        #     rank_loss2 = torch.mean(torch.where(new_scores2 < 0.0, torch.zeros_like(new_scores2), new_scores2))

                        #     rank_loss = 0.5*rank_loss1 + 0.5*rank_loss2
                        #     #rank_loss = 0.7*rank_loss1 + 0.3*rank_loss2
                            
                        if args.distill_list_num == 2:             
                            for pos in [0, 1]:            
                                sorted_indices_rest = torch.argsort(teacher_score[:, 2: ], descending=True)
                                #print(sorted_indices_rest)
                                sorted_indices = torch.cat([torch.full((scores.size(0), 1), pos, dtype=torch.long, device = args.device), sorted_indices_rest + 2], dim=1)
                                #print(sorted_indices)
                                sorted_score = torch.gather(scores, 1, sorted_indices)
                                #print(sorted_score)                            
                                sorted_scores.append(sorted_score)                            
                                
                            print(sorted_scores) 
                            if args.manual_rankNet:
                                print("-------manual_rankNet")
                                
                                rk_loss = []
                                for idx in [0,1]:
                                    num_scores = sorted_scores[idx].size(1)
                                    n = 100.
                                    # n = 200.
                                    #n = 300.
                                    #n = 400.
                                    print(n)
                                    first_row_weight = 3.
                                    # 使用 torch.meshgrid 生成所有成对组合的索引
                                    i_indices, j_indices = torch.meshgrid(torch.arange(num_scores, device=args.device), torch.arange(num_scores, device=args.device),indexing='ij')
                                    
                                    mask = i_indices < j_indices 
                                    
                                    diffs = sorted_scores[idx][:, j_indices] - sorted_scores[idx][:, i_indices] + args.rank_margin + (j_indices - i_indices - 1) * n
                                    
                                    diffs = torch.where(diffs < 0, torch.zeros_like(diffs), diffs)
                                    #print("diffs:",diffs)
                                    # weight for manual_rk
                                    A, B, C = diffs.size()
                                    weights = torch.ones((A, B, C), device=diffs.device)
                                    for i in range(B):
                                        if i == 0:
                                            weights[:, i, i+1:] = first_row_weight
                                        else:
                                            weights[:, i, i+1:] = 1 / (i + 1)
                                        
                                    weights = weights.to(args.device)
                                    #print("weight:", weights)
                                    weighted_diffs = diffs * weights * mask
                                    #print('weighted_diffs',weighted_diffs)
                                    rk_loss.append(torch.mean(weighted_diffs[weighted_diffs != 0]))

                                    # mask = mask.unsqueeze(0)
                                    
                                    # rk_loss.append(torch.mean(diffs[mask]))
                    
                            else:                                
                                
                                if args.sample_length == 6:
                                
                                    increments = torch.tensor([[0, 100, 200, 300],
                                                                [100, 200, 300, 400]])
                                elif args.sample_length == 8:
                                    increments = torch.tensor([[0, 100, 200, 300, 400, 500],
                                                                [100, 200, 300, 400, 500, 600]])
                                margin_teacher.append((args.rank_margin + increments[0]).to(args.device))
                                margin_teacher.append((args.rank_margin + increments[1]).to(args.device))
                                rk_loss = []
                                for idx in [0,1]:
                                    differences = sorted_scores[idx][:,1:] - sorted_scores[idx][:,0]
                                    #print("diff_score:", differences)            

                                    df_score = differences + margin_teacher[idx]
                                    print("differences----df_score:", differences, df_score)
                                    rk_loss.append(torch.mean(torch.where(df_score < 0.0,  torch.zeros_like(df_score), df_score)))
               
                            rank_loss1, rank_loss2 = rk_loss[0], rk_loss[1]
                            rank_loss = 0.5*rank_loss1 + 0.5*rank_loss2

                            #rank_loss = 0.3*rank_loss1 + 0.7*rank_loss2
                            #print(rk_loss)

                        if args.distill_list_num == 4:
                         
                            for pos in [0, 1,2,3]:            
                                sorted_indices_rest = torch.argsort(teacher_score[:, 4: ], descending=True)
                                #print(sorted_indices_rest)
                                sorted_indices = torch.cat([torch.full((scores.size(0), 1), pos, dtype=torch.long, device = args.device), sorted_indices_rest + 2], dim=1)
                                #print(sorted_indices)
                                sorted_score = torch.gather(scores, 1, sorted_indices)
                                #print(sorted_score)                            
                                sorted_scores.append(sorted_score)
                            increments = torch.tensor([[0, 100, 200, 300],
                                                        [100, 200, 300, 400],
                                                        [100, 200, 300, 400],
                                                        [100, 200, 300, 400]])
                            for i in range(4):
                                margin_teacher.append((args.rank_margin + increments[i]).to(args.device))
                            
                            rk_loss = []
                            for idx in [0,1,2,3]:
                                differences = sorted_scores[idx][:,1:] - sorted_scores[idx][:,0]
                                #print("diff_score:", differences)            

                                df_score = differences + margin_teacher[idx]
                                #print("differences----df_score:", differences, df_score)
                                rk_loss.append(torch.mean(torch.where(df_score < 0.0,  torch.zeros_like(df_score), df_score)))
               
                            rank_loss1, rank_loss2, rank_loss3, rank_loss4 = rk_loss[0], rk_loss[1],rk_loss[2], rk_loss[3]
                            rank_loss = 0.25*rank_loss1 + 0.25*rank_loss2 + 0.25*rank_loss3 + 0.25*rank_loss4
                            
                    # order negative passage by teacher, positve keeps idx=0
                    elif args.sort_negative_only:
                        print("-----sort_negative_only------")
                        sorted_indices_rest = torch.argsort(teacher_score[:, 1:], descending=True)
                        print(sorted_indices_rest)
                        sorted_indices = torch.cat([torch.zeros(scores.size(0), 1, dtype=torch.long, device=args.device), sorted_indices_rest + 1], dim=1)
                        print(sorted_indices)
                        sorted_score = torch.gather(scores, 1, sorted_indices)
                        print(sorted_score)
                        
                        new_scores1 = sorted_score[:,1] - sorted_score[:,0] + args.rank_margin
                        rank_loss1 = torch.mean(torch.where(new_scores1 < 0.0, torch.zeros_like(new_scores1), new_scores1))

                        new_scores2 = sorted_score[:,2] - sorted_score[:,0] + args.rank_margin + 100
                        rank_loss2 = torch.mean(torch.where(new_scores2 < 0.0, torch.zeros_like(new_scores2), new_scores2))

                        new_scores3 = sorted_score[:,3] - sorted_score[:,0] + args.rank_margin + 200
                        rank_loss3 = torch.mean(torch.where(new_scores3 < 0.0, torch.zeros_like(new_scores3), new_scores3))

                        new_scores4 = sorted_score[:,4] - sorted_score[:,0] + args.rank_margin + 300
                        rank_loss4 = torch.mean(torch.where(new_scores4 < 0.0, torch.zeros_like(new_scores4), new_scores4))
                        
                        rank_loss = 0.25*rank_loss1 + 0.25*rank_loss2 + 0.25*rank_loss3 + 0.25*rank_loss4
                    else:
                        sorted_indices = torch.argsort(teacher_score, descending=True)
                        #print("sorted_indices:  ", sorted_indices)
                        sorted_score = torch.gather(scores, 1, sorted_indices)
                        #print("sorted_score_gen", sorted_score)

                        new_scores1 = sorted_score[:,1] - sorted_score[:,0] + args.rank_margin
                        rank_loss1 = torch.mean(torch.where(new_scores1 < 0.0, torch.zeros_like(new_scores1), new_scores1))

                        new_scores2 = sorted_score[:,2] - sorted_score[:,0] + args.rank_margin + 100
                        rank_loss2 = torch.mean(torch.where(new_scores2 < 0.0, torch.zeros_like(new_scores2), new_scores2))

                        new_scores3 = sorted_score[:,3] - sorted_score[:,0] + args.rank_margin + 200
                        rank_loss3 = torch.mean(torch.where(new_scores3 < 0.0, torch.zeros_like(new_scores3), new_scores3))

                        new_scores4 = sorted_score[:,4] - sorted_score[:,0] + args.rank_margin + 300
                        rank_loss4 = torch.mean(torch.where(new_scores4 < 0.0, torch.zeros_like(new_scores4), new_scores4))
                        
                        rank_loss = 0.25*rank_loss1 + 0.25*rank_loss2 + 0.25*rank_loss3 + 0.25*rank_loss4
                        #rank_loss = rank_loss1
                        #print("new score", new_scores1, new_scores2, new_scores3, new_scores4)
                        #print("rank_loss", rank_loss)

                else:
                    

                    if args.loss_func == "listNet":
                        loss_function = listNet
                    elif args.loss_func == "listNet_add_norm":
                        loss_function = listNet_add_norm
                    elif args.loss_func == "KLloss":
                        loss_function = KLloss
                    # elif args.loss_func == "KLloss_add_Norm":
                    #     loss_function = KLloss_add_Norm
                    elif args.loss_func == "KLloss_add_Norm_fixed":
                        loss_function =  KLloss_add_Norm_fixed
                    elif args.loss_func == "rankNet":                            
                        loss_function = rankNet
                    elif args.loss_func ==  "rankNet_add_norm":
                        loss_function = rankNet_add_norm
                    elif args.loss_func == "lambdaLoss":            
                        loss_function = lambdaLoss
                    elif args.loss_func == "lambdaLoss_add_norm": 
                        loss_function = lambdaLoss_add_norm
                    #rank_loss = loss_function(scores, teacher_score, weighing_scheme="ndcgLoss2_scheme")
                    #rank_loss = loss_function(scores, teacher_score, weighing_scheme="ndcgLoss1_scheme")
                    rank_loss = loss_function(scores, teacher_score)

            #generation_loss = 0.33*generation_loss_span + 0.33*generation_loss_title + 0.33*generation_loss_query
            generation_loss = 0.5*generation_loss_span + 0.5*generation_loss_title
            generation_loss_query = 0

            if args.genLoss_only and not args.LTR_without_teacher:
                print("------generation_loss only------")
                loss = generation_loss
            else:
                loss =  args.factor_of_rank_loss*rank_loss+args.factor_of_generation_loss*generation_loss

            if args.LTR_without_teacher or not args.rest_all_negative:
                if args.distill_list_num == 4:
                    return loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query, generation_loss, rank_loss1, rank_loss2, rank_loss3, rank_loss4
                return loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query, generation_loss, rank_loss1, rank_loss2
                
            elif args.order_margin and args.rest_all_negative:
                return loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query, generation_loss, rank_loss1, rank_loss2, rank_loss3, rank_loss4
            else:
                return loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query, generation_loss

from torch.distributed.elastic.multiprocessing.errors import record

@record
def train(args, model, tokenizer):
    output_dir = os.path.join(
        #args.output_dir, f"{args.teacher_kind}_{args.loss_func}_{args.scenario_name}/{args.factor_of_generation_loss}")
        args.output_dir, f"{args.scenario_name}/{args.factor_of_generation_loss}_{args.num_train_epochs}_{args.experiment_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    
    DatasetClass = FinetuningDataset
    train_dataset = DatasetClass(args.train_file, tokenizer, args.load_small, top_p=args.top_p, top_ngram=args.top_ngram, margin=args.rank_margin,pid2query=args.pid2query, 
                                 sample_length= args.sample_length, shuffle_positives = args.shuffle_positives,shuffle_negatives = args.shuffle_negatives, use_top10 = args.use_top10, 
                                 distill_list_num =args.distill_list_num, 
                                 teacher_kind =args.teacher_kind, rest_all_negative= args.rest_all_negative, LTR_without_teacher = args.LTR_without_teacher)
   

    run_dir = Path(f"{os.getcwd()}/experiment_logs_same_sample_NQ") / args.scenario_name/ args.experiment_name
    #run_dir = Path(f"{os.getcwd()}/experiment_logs_TriviaQA") / args.scenario_name/ args.experiment_name
   
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    #name= args.experiment_name+"_"+str(args.factor_of_generation_loss),
    print(args.local_rank)
    if args.local_rank == 0:
        wandb.init(config=args,
                project=args.project_name,
                notes=socket.gethostname(),
                group= args.scenario_name,
                dir=str(run_dir),
                name= args.experiment_name+"_"+str(args.factor_of_generation_loss),
                job_type = args.experiment_name+"_"+str(args.factor_of_generation_loss),
                reinit=True)
        wandb.watch(model)
    
    # early stopping
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    early_stop = False
    
    
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(output_dir, 'logs'))


    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_sampler = SequentialSampler(
    #         train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers) #, pin_memory = True, prefetch_factor= 1

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)


    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


   # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps == 0:
        args.warmup_steps = int(t_total * args.warmup_portion)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, rk_loss, gen_loss = 0.0, 0.0, 0.0
    logging_loss, logging_loss_rk, logging_loss_gen = 0.0, 0.0, 0.0
    rk_loss1, rk_loss2, rk_loss3, rk_loss4 =  0.0, 0.0, 0.0, 0.0
    logging_loss_rk1, logging_loss_rk2, logging_loss_rk3, logging_loss_rk4 = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    ntokens = float(searcher.fm_index.beginnings[-1])
    global_step_list = []
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(epoch_iterator):

            if args.fp16:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    
                    if args.order_margin and args.rest_all_negative:
                        loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query,generation_loss, rank_loss1, rank_loss2, rank_loss3, rank_loss4 = model_forward(args,batch,model,ntokens)
                    elif args.LTR_without_teacher or not args.rest_all_negative:
                        if args.distill_list_num == 4:
                            loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query, generation_loss, rank_loss1, rank_loss2, rank_loss3, rank_loss4 = model_forward(args,batch,model,ntokens)
                        else:
                            loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query,generation_loss, rank_loss1, rank_loss2= model_forward(args,batch,model,ntokens)
                    else:
                        loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query,generation_loss = model_forward(args,batch,model,ntokens)
            else:
                    
                    if args.order_margin and args.rest_all_negative:
                        loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query,generation_loss, rank_loss1, rank_loss2, rank_loss3, rank_loss4 = model_forward(args,batch,model,ntokens)
                    elif args.LTR_without_teacher or not args.rest_all_negative:
                        if args.distill_list_num == 4:
                            loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query, generation_loss, rank_loss1, rank_loss2, rank_loss3, rank_loss4 = model_forward(args,batch,model,ntokens)
                        else:
                            loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query,generation_loss, rank_loss1, rank_loss2= model_forward(args,batch,model,ntokens)
                    else:
                        loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query,generation_loss = model_forward(args,batch,model,ntokens)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                rank_loss = rank_loss /  args.gradient_accumulation_steps
                generation_loss = generation_loss / args.gradient_accumulation_steps
                
                
                if args.LTR_without_teacher or not args.rest_all_negative:
                    rank_loss1 = rank_loss1 /  args.gradient_accumulation_steps
                    rank_loss2 = rank_loss2 /  args.gradient_accumulation_steps
                    if args.distill_list_num == 4:
                        rank_loss3 = rank_loss3 /  args.gradient_accumulation_steps
                        rank_loss4 = rank_loss4 /  args.gradient_accumulation_steps
                if args.order_margin and args.rest_all_negative:
                    rank_loss1 = rank_loss1 /  args.gradient_accumulation_steps
                    rank_loss2 = rank_loss2 /  args.gradient_accumulation_steps
                    rank_loss3 = rank_loss3 /  args.gradient_accumulation_steps
                    rank_loss4 = rank_loss4 /  args.gradient_accumulation_steps

            
            if args.fp16:
                #print("------loss:", loss)
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            rk_loss += rank_loss.item()
            gen_loss += generation_loss.item()
            if args.LTR_without_teacher or not args.rest_all_negative:
                rk_loss1 += rank_loss1.item()
                rk_loss2 += rank_loss2.item()
                if args.distill_list_num == 4:
                    rk_loss3 += rank_loss3.item()
                    rk_loss4 += rank_loss4.item()
            if args.order_margin and args.rest_all_negative:
                rk_loss1 += rank_loss1.item()
                rk_loss2 += rank_loss2.item()
                rk_loss3 += rank_loss3.item()
                rk_loss4 += rank_loss4.item()
            

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()

                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                if args.local_rank in [-1, 0]:
                    print('loss', loss.item())
                    print('rank_loss', rank_loss.item())
                    print('generation_loss', generation_loss.item())
                    # print('generation_loss_title', generation_loss_title.item())
                    # print('generation_loss_span', generation_loss_span.item())
                    # print('generation_loss_query', generation_loss_query.item())
                    if args.LTR_without_teacher or not args.rest_all_negative:
                        print('rank_loss1', rank_loss1.item())
                        print('rank_loss2', rank_loss2.item())
                        if args.distill_list_num == 4:
                            print('rank_loss3', rank_loss3.item())
                            print('rank_loss4', rank_loss4.item())
                    if args.order_margin and args.rest_all_negative:
                        print('rank_loss1', rank_loss1.item())
                        print('rank_loss2', rank_loss2.item())
                        print('rank_loss3', rank_loss3.item())
                        print('rank_loss4', rank_loss4.item())
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rank_loss', (rk_loss - logging_loss_rk)/ args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'generation_loss', (gen_loss - logging_loss_gen)/ args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rkloss1', (rk_loss1 - logging_loss_rk1)/ args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rkloss2', (rk_loss2 - logging_loss_rk2)/ args.logging_steps, global_step)
                    

                    
                    wandb.log({'loss': (tr_loss - logging_loss)/args.logging_steps}, step=global_step)
                    wandb.log({'rank_loss': (rk_loss - logging_loss_rk)/ args.logging_steps} ,step=global_step)
                    wandb.log({'generation_loss': (gen_loss - logging_loss_gen)/ args.logging_steps} ,step=global_step)
                    wandb.log({'lr': scheduler.get_lr()[0]} ,step=global_step)
                    if args.LTR_without_teacher or not args.rest_all_negative:
                        wandb.log({'rkloss1': (rk_loss1 - logging_loss_rk1)/args.logging_steps}, step=global_step)
                        wandb.log({'rkloss2': (rk_loss2 - logging_loss_rk2)/args.logging_steps}, step=global_step)
                        if args.distill_list_num == 4:
                            wandb.log({'rkloss3': (rk_loss3 - logging_loss_rk3)/args.logging_steps}, step=global_step)
                            wandb.log({'rkloss4': (rk_loss4 - logging_loss_rk4)/args.logging_steps}, step=global_step)
                    if args.order_margin and args.rest_all_negative:
                        wandb.log({'rkloss1': (rk_loss1 - logging_loss_rk1)/args.logging_steps}, step=global_step)
                        wandb.log({'rkloss2': (rk_loss2 - logging_loss_rk2)/args.logging_steps}, step=global_step)
                        wandb.log({'rkloss3': (rk_loss3 - logging_loss_rk3)/args.logging_steps}, step=global_step)
                        wandb.log({'rkloss4': (rk_loss4 - logging_loss_rk4)/args.logging_steps}, step=global_step)
   
                    logging_loss = tr_loss
                    logging_loss_rk = rk_loss
                    logging_loss_gen = gen_loss

                    if args.LTR_without_teacher or not args.rest_all_negative:
                        logging_loss_rk1 = rk_loss1
                        logging_loss_rk2 = rk_loss2
                        if args.distill_list_num == 4:
                            logging_loss_rk3 = rk_loss3
                            logging_loss_rk4 = rk_loss4
                    elif args.order_margin and args.rest_all_negative:
                        logging_loss_rk1 = rk_loss1
                        logging_loss_rk2 = rk_loss2
                        logging_loss_rk3 = rk_loss3
                        logging_loss_rk4 = rk_loss4
                    
                    if args.use_early_stopping:
                        val_loss = loss.item()
                        if  val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print(f"Early stopping triggered. Stopping training at epoch {epoch}")
                            early_stop = True                            
                            break
                
                if args.save_steps != -1 and global_step % args.save_steps == 0 or early_stop:
                    global_step_list.append(global_step)
                    if args.local_rank in [-1, 0]:

                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        
                        # Save model checkpoint
                        checkpoints_output_dir = os.path.join(
                            output_dir, 'checkpoint-{}'.format(global_step))
                        #output_dir = os.path.join(args.output_dir, args.teacher_kind, args.experiment_name, 'checkpoint-{}'.format(global_step))
                        torch.save(model_to_save.state_dict(), checkpoints_output_dir+"model.pt")
                        logger.info("Saving model checkpoint to %s", checkpoints_output_dir)

            #             str=('TOKENIZERS_PARALLELISM=false python -m seal.search \
            # --topics_format dpr_qas --topics /project/E2ESEAL/data/NQ/nq-test.qa.csv \
            # --output_format dpr --output output_test.json \
            # --checkpoint ' + output_dir+"model.pt" + ' \
            # --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
            # --decode_query no --fm_index /project/E2ESEAL/data/SEAL-checkpoint+index.NQ/NQ.fm_index --dont_fairseq_checkpoint')
            #             p=os.system(str)
            #             print(p) 
            #             str=('python3 evaluate_output.py --file output_test.json')
            #             p=os.system(str)
            #             print(p) 

        if args.save_steps > -2 or early_stop:
            global_step_list.append(global_step)
            if args.local_rank in [-1, 0]:

                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                # Save model checkpoint
                # output_dir = os.path.join(
                #      output_dir,'checkpoint-{}'.format(global_step))
                
                # torch.save(model_to_save.state_dict(), output_dir+"model.pt")
                # logger.info("Saving model checkpoint to %s", output_dir)
                checkpoints_output_dir = os.path.join(
                            output_dir, 'checkpoint-{}'.format(global_step))                        
                torch.save(model_to_save.state_dict(), checkpoints_output_dir+"model.pt")
                logger.info("Saving model checkpoint to %s", checkpoints_output_dir)
            #     str=('TOKENIZERS_PARALLELISM=false python -m seal.search \
            # --topics_format dpr_qas --topics /project/E2ESEAL/data/NQ/nq-test.qa.csv \
            # --output_format dpr --output output_test.json \
            # --checkpoint ' + output_dir+"model.pt" + ' \
            # --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
            # --decode_query no --fm_index /project/E2ESEAL/data/SEAL-checkpoint+index.NQ/NQ.fm_index --dont_fairseq_checkpoint')
            #     p=os.system(str)
            #     print(p) 
            #     str=('python3 evaluate_output.py --file output_test.json')
            #     p=os.system(str)
            #     print(p) 
    if args.local_rank in [-1, 0]:
        wandb.finish()
    return global_step, tr_loss / global_step, global_step_list


parser = argparse.ArgumentParser()

parser.add_argument("--shuffle_positives", default=False, type=str2bool, required=False,
                    help="whether to shffle a postive passage to a query")
parser.add_argument("--shuffle_negatives", default=False, type=str2bool, required=False,
                    help="whether to shffle a negative passage to a query")
parser.add_argument('--checkpoint', default="minder_ltr/checkpoint_minder_NQ.pt", required=False, type=str)
parser.add_argument('--fm_index', default="/minder_ltr/data/fm_index/psgs_w100.fm_index", required=False, type=str)
parser.add_argument("--top_p", default=4, type=int, required=False,
                    help="top p passages for training")
parser.add_argument("--top_ngram", default=40, type=int, required=False,
                    help="top ngrams for training")
parser.add_argument("--do_fm_index", default=True, type=str2bool,
                    help="whether to consider the fm index score or not")
parser.add_argument("--rank_margin", default=300, type=int, required=False,
                    help="the margin for the rank loss")
parser.add_argument("--factor_of_generation_loss", default=1000., type=float, required=False,
                    help="the factor to * the generation_loss")
parser.add_argument("--factor_of_rank_loss", default=1., type=float, required=False,
                    help="the factor to * the generation_loss")
parser.add_argument("--decode_query", default="stable",
                    type=str, required=True)
# data
#"/minder_ltr/data/NQ/nq-test.csv“
parser.add_argument('--pid2query', default="/minder_ltr/data/pid2query.pkl", type=str)
parser.add_argument("--train_file", default="/minder_ltr/test_input/test_loss.json",
                    type=str, required=False,
                    help="training file ")
parser.add_argument("--dev_file", default="//project/E2ESEAL/code/SEAL/ngrams_train.json",
                    type=str, required=False,
                    help="dev_file ")
parser.add_argument("--test_file", default="/project/E2ESEAL/data/NQ/nq-test.qa.csv",
                    type=str, required=False,
                    help="test_file ")
# parser.add_argument("--output_dir", default='./checkpoints_logs/release_test_simLM_3epoch', type=str, required=False,
#                     help="The output directory where the model checkpoints and predictions will be written.")
# parser.add_argument("--output_dir", default='./experiment_chechpoint_TriviaQA', type=str, required=False,
#                     help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--output_dir", default='./experiment_data_same_sample_NQ', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")                    
parser.add_argument("--load_small", default=False, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=0, type=int, required=False,
                    help="number of workers for dataloader")

# training
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_test", default=False, type=str2bool,
                    help="Whether to run eval on the test set.")



parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--rescore_batch_size", default=50, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--per_gpu_test_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for evaluation.")

parser.add_argument("--learning_rate", default=1e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=1, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument('--do_lower_case', type=str2bool, default=True,
                    help="tokenizer do_lower_case")


parser.add_argument('--logging_steps', type=int, default=1,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=-1,
                    help="Save checkpoint every X updates steps.")

parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=True, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision")
parser.add_argument("--sample_length", default = 5, type  = int, 
                    help = "passage number per query")
parser.add_argument("--teacher_kind", default="score_teacher_E5", type = str)
parser.add_argument("--rest_all_negative",type=str2bool, default=True, 
                    help = "sampel strategy, whether all sample are negative except the highest ones")
parser.add_argument("--loss_func", default="listNet_add_norm", type = str)
#wandb config
#parser.add_argument("--team_name", type=str)
parser.add_argument("--project_name", type=str, default = "LTR-listwise")
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--scenario_name", type=str, default = "factor")
parser.add_argument("--genLoss_only", type=str2bool,default=False,
                    help ="use only generation loss to train, without rank loss")
parser.add_argument("--LTR_without_teacher",type=str2bool , default=False, 
                    help = "reproduce LTR v.14 without teacher")
parser.add_argument('--use_early_stopping', default = False, 
                    help='Enable early stopping strategy')
parser.add_argument('--order_margin', type=str2bool, default = False, 
                    help='construct loss using teacher score order')

parser.add_argument('--use_LTR_checkpoint', type=str2bool, default = False, 
                    help='train based on LTR_checkpoint')
parser.add_argument('--sort_negative_only',type=str2bool, default = False, 
                    help = 'compute central margin loss only order negative by teacher' )
parser.add_argument('--use_top10',type=str2bool, default = False, 
                    help = 'use top10 score passage' )
parser.add_argument('--manual_rankNet',type=str2bool, default = False, 
                    help = 'use top10 score passage' )
parser.add_argument("--distill_list_num", default = 2, type  = int, 
                    help = "distill_list_num")
# parser.add_argument('--use_two_positive',type=str2bool, default = False, 
#                     help = 'compute central margin loss but use two_positive' )




# #多gpu train loss
# args, unknown = parser.parse_known_args()

# if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
#     raise ValueError(
#         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

# torch.distributed.init_process_group(backend='nccl')

# # 配置每个进程的gpu
# local_rank = torch.distributed.get_rank()
# args.local_rank=local_rank
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
# args.device = device

# # Setup logging
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
# logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
#                args.local_rank, device, bool(args.local_rank != -1), args.fp16)

#原代码
args, unknown = parser.parse_known_args()
# if args.local_rank!=-1:
args.local_rank = int(os.environ["LOCAL_RANK"])
if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
print("----------",args.local_rank)
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)


print(args.device)



searcher = SEALSearcher.load(args.fm_index, args.checkpoint,device=args.device, decode_query=args.decode_query)


logger.info("Training/evaluation parameters %s", args)
if args.do_train:
    # run_dir = Path(f"{os.getcwd()}/wandb_locals") / args.scenario_name/ args.experiment_name
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    # wandb.init(config=args,
    #             project=args.project_name,
    #             notes=socket.gethostname(),
    #             name= args.experiment_name+"_"+str(args.factor_of_generation_loss),
    #             group= args.scenario_name,
    #             dir=str(run_dir),
    #             reinit=True)
    # wandb.watch(searcher.bart_model)


    global_step, tr_loss, global_step_list = train(
        args, searcher.bart_model.to(args.device), searcher.bart_tokenizer)
    # wandb.finish()
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)


# with wandb.init(config=vars(hyperparameters),
#                         project=hyperparameters.project_name,
#                         group=hyperparameters.scenario_name,
#                         name=hyperparameters.experiment_name+"_"+str(seed),
#                         notes=hyperparameters.note, 
#                         dir=run_dir):
# ————————————————
# 版权声明：本文为CSDN博主「云端FFF」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/wxc971231/article/details/128941933