# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from collections import defaultdict
import json
import multiprocessing
import random
import re
import tqdm
import math

import ftfy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import nltk
import pickle
from datasets import load_dataset
nltk.download('stopwords')
banned = set(stopwords.words('english'))

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--min_score', default=999.0, type=float)
    parser.add_argument('--min_score_gold', default=999.0, type=float)
    parser.add_argument('--max_rank', default=1, type=int)
    parser.add_argument(
        '--target',
        default = "span",
        choices = [
            "chunk",
            "span",
            "title",
            "code",
            "query",
        ])
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--jobs', default=30, type=int)
    parser.add_argument('--mark_target', action="store_true")
    parser.add_argument('--mark_silver', action="store_true")
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--id2code', default=None, type=str)
    parser.add_argument('--pid2query', default=None, type=str)

    parser.add_argument('--mode', choices=["w", "a"], default="w")


    return parser.parse_args()

def read_id2code(id2code_path):

    id2code = {}

    with open(id2code_path) as fin:

        for line in tqdm.tqdm(fin):
            line = line.strip()
            if not line:
                continue
            idx, code = line.split("\t")
            id2code[idx] = code

    return id2code

def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i+ngrams)
        # for j in range(i+1, min(i+1+ngrams, len(tokens) + 1)):
        #     tok_orig = tokens[i:j]
        #     tok = [t for t in tok_orig if t not in banned]
        #     if not tok:
        #         break
        #     yield (i, j)

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

    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i+subspan_size])
        yield span

def extract_spans_wrapper(args):
    return args[1], list(extract_spans(*args))

def clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = ftfy.fix_text(text)
    text = text.replace('BULLET::::', '')
    text = text.replace('SECTION::::', '')
    text = text.strip()
    return text

def _iterator_span_get_arguments(data, min_score, max_rank, mark_target, mark_silver, min_score_gold):
  
    for sample in tqdm.tqdm(data):
        
        source = sample['query'].strip()
        if mark_target:
            source += " || body"   
        
        for i, ctx in enumerate(sample['positive_passages'], start=1):
            ctx['score']=1000.0
            if i > max_rank:
                continue

            if float(ctx['score']) < min_score:
                continue
             
            text = ctx['text'].strip()
            if mark_silver and float(ctx['score']) < min_score_gold:
                yield text, source + " || ?"
            elif mark_silver:
                yield text, source + " || +"
            else:
                yield text, source
            
def iterator_span(args):
    if "train" in args.output:
        data = load_dataset(args.input, split="train")

    if "dev" in args.output:
        qid2query={}
        for s in load_dataset("Tevatron/msmarco-passage", split="dev"):
            qid2query[s['query_id']] = s['query']
        passage_corpus = load_dataset("Tevatron/msmarco-passage-corpus", split="train")

        data = []
        with open("/home/v-yongqili/project/GGR/data/MSMARCO/qrels.msmarco-passage.dev-subset.txt",'r') as f:
            for l in f:
                entry = {}

                l = re.split('[\t\s]', l.strip())
                qid = str(l[0])
                entry['query_id'] = qid
                entry['query'] = qid2query[qid]
                entry['positive_passages'] = [passage_corpus[int(l[2])]]

                data.append(entry)

    arg_it = _iterator_span_get_arguments(data, args.min_score, args.max_rank, args.mark_target, args.mark_silver, args.min_score_gold)
    arg_it = ((text, source, args.n_samples, args.min_length, args.max_length, args.temperature) for text, source in arg_it)

    with multiprocessing.Pool(args.jobs) as pool:
        for source, spans in pool.imap(extract_spans_wrapper, arg_it):
            for target in spans:
                yield source, target

def iterator(args):
    if args.pid2query and args.target == "query":
        with open(args.pid2query, 'rb') as f:
            pid2query = pickle.load(f)

    if args.target == "code" and args.id2code:
        id2code = read_id2code(args.id2code)

    if "train" in args.output:
        data = load_dataset(args.input, split="train")

    if "dev" in args.output:
        qid2query={}
        for s in load_dataset("Tevatron/msmarco-passage", split="dev"):
            qid2query[s['query_id']] = s['query']
        passage_corpus = load_dataset("Tevatron/msmarco-passage-corpus", split="train")

        data = []
        with open("/home/v-yongqili/project/GGR/data/MSMARCO/qrels.msmarco-passage.dev-subset.txt",'r') as f:
            for l in f:
                entry = {}

                l = re.split('[\t\s]', l.strip())
                qid = str(l[0])
                entry['query_id'] = qid
                entry['query'] = qid2query[qid]
                entry['positive_passages'] = [passage_corpus[int(l[2])]]

                data.append(entry)
    for sample in tqdm.tqdm(data):

        source = sample['query'].strip()



        if args.target == "chunk" and args.mark_target:
            source += " || body"    
            
        elif args.target == "title" and args.mark_target:
            source += " || title"    
        elif args.target == "query" and args.mark_target:
            source += " || query"                
        elif args.target == "code" and args.mark_target:
            source += " || code"    

        else:
            raise ValueError("Wrong target")

        for i, ctx in enumerate(sample['positive_passages'], start=1):

            ctx['score']=1000.0

            if i > args.max_rank:
                continue
            if float(ctx['score']) < args.min_score:
                continue

            if args.target == "chunk":
                target = ctx['text'].strip()
                for _ in range(args.n_samples):
                    if args.mark_silver and float(ctx['score']) < args.min_score_gold:
                        yield source + " || ?", target
                    elif args.mark_silver:
                        yield source + " || +", target
                    else:
                        yield source, target
            

            elif args.target == "title":
                if ctx['title'].strip()=="-":
                    continue
                target = ctx['title'].strip() + " @@"

                for _ in range(args.n_samples):
                    if args.mark_silver and float(ctx['score']) < args.min_score_gold:
                        yield source + " || ?", target
                    elif args.mark_silver:
                        yield source + " || +", target
                    else:
                        yield source, target
            elif args.target == "query":

                if "trivia" in args.input:
                    querys = pid2query[str(ctx['psg_id'])]
                if "nq" in args.input:
                    querys = pid2query[str(ctx['passage_id'])]
                if "msmarco" in args.input:
                    querys = pid2query[str(ctx['docid'])]
                for _ in range(args.n_samples):
                    if args.mark_silver and float(ctx['score']) < args.min_score_gold:
                        yield source + " || ?", querys[min(_, len(querys)-1)]
                    elif args.mark_silver:
                        yield source + " || +", querys[min(_, len(querys)-1)]
                    else:
                        yield source, querys[min(_, len(querys)-1)]
            elif args.target == "code":
                idx = ctx['docid']
                code = id2code.get(idx)
                if not code: continue
                target = code.strip() + " ||"
                for _ in range(args.n_samples):
                    if args.mark_silver and float(ctx['score']) < args.min_score_gold:
                        yield source + " || ?", target
                    elif args.mark_silver:
                        yield source + " || +", target
                    else:
                        yield source, target

            else:
                raise ValueError("Wrong target")

def main():

    args = parse_args()
    with open(args.output + '.source', mode=args.mode) as src, open(args.output + '.target', mode=args.mode) as tgt:

        for source, target in iterator_span(args) if args.target == "span" else iterator(args):
            source = " " + source.strip()
            target = " " + target.strip()

            src.write(source + "\n")
            tgt.write(target + "\n")

if __name__ == '__main__':

    main()
