# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import os


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
nltk.download('stopwords')
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
    def __init__(self, filename, tokenizer,
                 load_small, shuffle_positives, shuffle_negatives, top_p, top_ngram, margin, pid2query):
        

        self._load_small = load_small

        self.shuffle_positives = shuffle_positives
        self.shuffle_negatives = shuffle_negatives
        self.top_p = top_p
        self.top_ngram = top_ngram
        self.tokenizer = tokenizer
        self.margin = margin

        self.data = []
        if pid2query:
            with open(pid2query, 'rb') as f:
                self.pid2query = pickle.load(f)
        if os.path.isfile(filename):
            with open(filename, 'r') as load_f:
                for line in tqdm(load_f):
                    line = json.loads(line)
                    if len(line['positive_ctxs'])>0  and len(line['negative_ctxs']) > 0:
                        if line['positive_ctxs'][0]['score'] < line['negative_ctxs'][0]['score']+self.margin :
                            self.data.append(line)    

        else:
            for file in os.listdir(filename):
                with open(filename+file, 'r') as load_f:
                    for line in tqdm(load_f):
                        line = json.loads(line)
                        if len(line['positive_ctxs'])>0 and len(line['negative_ctxs'])>0:
                            if line['positive_ctxs'][0]['score'] < line['negative_ctxs'][0]['score']+self.margin :
                                self.data.append(line)       

        self._total_data = 0
        if self._load_small:
            self._total_data = 100
        else:
            self._total_data = len(self.data)

    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
  
        entry = self.data[idx]

        positive_ctxs = entry['positive_ctxs']
        negative_ctxs = entry['negative_ctxs']

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
                    found_keys.append([0.0, key[0], 10000])


        return_feature_dict = {   'question': entry['question'], 
                                  'found_keys': found_keys,
                                  'title_target':title_target,
                                  'span_target':span_target,
                                  'query_target':query_target,
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
                found_keys=batch['found_keys']
                title_target=batch['title_target']
                span_target=batch['span_target']
                query_target=batch['query_target']

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
                            query_index.append(0)
                        elif k[0]== 45056:
                            title_index.append(0)
                            body_index.append(0)
                            query_index.append(1)
                        else:
                            title_index.append(0)
                            body_index.append(1)
                            query_index.append(0)  
                        ngrams_len.append(len(k))  
                        tts = set(k)
                        coeff.append(repetition(tts, current_coverage))
                        current_coverage |= tts 
                        i+=1      

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


            scores = scores_title*torch.Tensor(title_index).to(scores_title.device) + scores_body*torch.Tensor(body_index).to(scores_title.device) +  scores_query*torch.Tensor(query_index).to(scores_query.device)

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


            scores = torch.sum(scores, dim=1)

            scores = torch.reshape(scores, (-1, args.top_p))

            new_scores1 = scores[:,1] - scores[:,0] + args.rank_margin
            # print('postive',scores[:,0])
            # print('negative',scores[:,1])
            rank_loss1 = torch.mean(torch.where(new_scores1 < 0.0, torch.zeros_like(new_scores1), new_scores1))

            new_scores2 = scores[:,3] - scores[:,2] + args.rank_margin + 100
            # print('postive',scores[:,0])
            # print('negative',scores[:,1])
            rank_loss2 = torch.mean(torch.where(new_scores2 < 0.0, torch.zeros_like(new_scores2), new_scores2))

            rank_loss = 0.5*rank_loss1 + 0.5*rank_loss2
            ####################
            generation_loss = 0.33*generation_loss_span + 0.33*generation_loss_title + 0.33*generation_loss_query

            loss =  rank_loss+args.factor_of_generation_loss*generation_loss

            return loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query

def train(args, model, tokenizer):
    DatasetClass = FinetuningDataset
    train_dataset = DatasetClass(args.train_file, tokenizer,
                                 args.load_small,
                                 shuffle_positives=args.shuffle_positives, 
                                 shuffle_negatives=args.shuffle_negatives,
                                 top_p=args.top_p, top_ngram=args.top_ngram, margin=args.rank_margin,pid2query=args.pid2query)

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))


    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)


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
    tr_loss, logging_loss = 0.0, 0.0
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
                    loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query = model_forward(args,batch,model,ntokens)
            else:
                    loss,rank_loss,generation_loss_title,generation_loss_span,generation_loss_query = model_forward(args,batch,model,ntokens)


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tr_loss += loss.item()
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
                    print('generation_loss_title', generation_loss_title.item())
                    print('generation_loss_span', generation_loss_span.item())
                    print('generation_loss_query', generation_loss_query.item())
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps != -1 and global_step % args.save_steps == 0:
                    global_step_list.append(global_step)
                    if args.local_rank in [-1, 0]:

                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        # Save model checkpoint
                        output_dir = os.path.join(
                            args.output_dir, 'checkpoint-{}'.format(global_step))
                        torch.save(model_to_save.state_dict(), output_dir+"model.pt")
                        logger.info("Saving model checkpoint to %s", output_dir)

            #             str=('TOKENIZERS_PARALLELISM=false python -m seal.search \
            # --topics_format dpr_qas --topics /home/v-yongqili/project/E2ESEAL/data/NQ/nq-test.qa.csv \
            # --output_format dpr --output output_test.json \
            # --checkpoint ' + output_dir+"model.pt" + ' \
            # --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
            # --decode_query no --fm_index /home/v-yongqili/project/E2ESEAL/data/SEAL-checkpoint+index.NQ/NQ.fm_index --dont_fairseq_checkpoint')
            #             p=os.system(str)
            #             print(p) 
            #             str=('python3 evaluate_output.py --file output_test.json')
            #             p=os.system(str)
            #             print(p) 

        if args.save_steps > -2:
            global_step_list.append(global_step)
            if args.local_rank in [-1, 0]:

                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, 'checkpoint-{}'.format(global_step))
                torch.save(model_to_save.state_dict(), output_dir+"model.pt")
                logger.info("Saving model checkpoint to %s", output_dir)
            #     str=('TOKENIZERS_PARALLELISM=false python -m seal.search \
            # --topics_format dpr_qas --topics /home/v-yongqili/project/E2ESEAL/data/NQ/nq-test.qa.csv \
            # --output_format dpr --output output_test.json \
            # --checkpoint ' + output_dir+"model.pt" + ' \
            # --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
            # --decode_query no --fm_index /home/v-yongqili/project/E2ESEAL/data/SEAL-checkpoint+index.NQ/NQ.fm_index --dont_fairseq_checkpoint')
            #     p=os.system(str)
            #     print(p) 
            #     str=('python3 evaluate_output.py --file output_test.json')
            #     p=os.system(str)
            #     print(p) 
    return global_step, tr_loss / global_step, global_step_list


parser = argparse.ArgumentParser()

parser.add_argument("--shuffle_positives", default=False, type=str2bool, required=False,
                    help="whether to shffle a postive passage to a query")
parser.add_argument("--shuffle_negatives", default=False, type=str2bool, required=False,
                    help="whether to shffle a negative passage to a query")
parser.add_argument('--checkpoint', default="/home/v-yongqili/project/E2ESEAL/data/SEAL-checkpoint+index.NQ/SEAL.NQ.pt", required=False, type=str)
parser.add_argument('--fm_index', default="/home/v-yongqili/project/E2ESEAL/data/SEAL-checkpoint+index.NQ/NQ.fm_index", required=False, type=str)
parser.add_argument("--top_p", default=4, type=int, required=False,
                    help="top p passages for training")
parser.add_argument("--top_ngram", default=40, type=int, required=False,
                    help="top ngrams for training")
parser.add_argument("--do_fm_index", default=False, type=str2bool,
                    help="whether to consider the fm index score or not")
parser.add_argument("--rank_margin", default=300, type=int, required=False,
                    help="the margin for the rank loss")
parser.add_argument("--factor_of_generation_loss", default=1000, type=int, required=False,
                    help="the factor to * the generation_loss")
parser.add_argument("--decode_query", default="stable",
                    type=str, required=True)
# data
parser.add_argument('--pid2query', default=None, type=str)
parser.add_argument("--train_file", default="/home/v-yongqili/project/E2ESEAL/data/SEAL_on_train_transformed/",
                    type=str, required=False,
                    help="training file ")
parser.add_argument("--dev_file", default="/home/v-yongqili/project/E2ESEAL/code/SEAL/ngrams_train.json",
                    type=str, required=False,
                    help="dev_file ")
parser.add_argument("--test_file", default="/home/v-yongqili/project/E2ESEAL/data/NQ/nq-test.qa.csv",
                    type=str, required=False,
                    help="test_file ")
parser.add_argument("--output_dir", default='./release_test1', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--load_small", default=False, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=4, type=int, required=False,
                    help="number of workers for dataloader")

# training
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_test", default=False, type=str2bool,
                    help="Whether to run eval on the test set.")



parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
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


args, unknown = parser.parse_known_args()
if args.local_rank!=-1:
    args.local_rank = int(os.environ["LOCAL_RANK"])
if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

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

searcher = SEALSearcher.load(args.fm_index, args.checkpoint, device=args.device, decode_query=args.decode_query)



logger.info("Training/evaluation parameters %s", args)
if args.do_train:
    global_step, tr_loss, global_step_list = train(
        args, searcher.bart_model.to(args.device), searcher.bart_tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

