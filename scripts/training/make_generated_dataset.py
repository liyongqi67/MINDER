# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import random
import tqdm
import pickle
from nltk.corpus import stopwords
from random import sample
from fuzzywuzzy import fuzz
from collections import defaultdict
import math
banned = {
    "the", "The",
    "to", 
    "a", "A", "an", "An", 
    "he", "He", "his", "His", "him", "He's",  
    "she", "She", "her", "Her", "she's", "She's", 
    "it", "It", "its", "Its",  "it's", "It's",
    "and", "And",
    "or", "Or",
    "this", "This",
    "that", "That",
    "those", "Those",
    "these", "These",
    '"', '""', "'", "''",
}

def is_good(token):
    if token in banned:
        return False
    elif token[-1] in '?.!':
        return False
    elif token[0] in '([':
        return False
    return True
def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i+ngrams)

def extract_spans(text, source, n_samples, min_length=10, max_length=10, temperature=1.0):
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
    spans = []
    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i+subspan_size])
        spans.append(span)
    return spans


def preprocess_file(
    input_path,
    num_samples=1,
    num_title_samples=1,
    num_query_samples=1,
    format="dpr", 
    delimiter='@@', 
    min_length_input=1,
    max_length_input=15,
    min_length_output=10, 
    max_length_output=10,
    full_doc_n=0,
    mark_pretraining=False,
    pid2query=None
    ):
    
    if format == 'kilt':
        raise NotImplementedError
    elif format == 'dpr':
        if pid2query:
            with open(pid2query, 'rb') as f:
                pid2query = pickle.load(f)
        with open(input_path, 'r', 2 ** 20) as f:
            next(f)
            
            f = csv.reader(f, delimiter='\t', quotechar='"')
            f = (l for l in f if len(l) == 3)
            
            for pid, text, title in tqdm.tqdm(f):
                try:
                    text = text
                    title = title
                    querys = pid2query[str(pid)]


                    if text == title:
                        continue

                    sampled = 0
                    select_query = sample(querys, 1)[0]
                    a = select_query+ " || body"
                    if mark_pretraining:
                        a += " || p"
                    spans = extract_spans(text, a, num_samples, min_length=10, max_length=10, temperature=1.0)
                    while sampled < min(num_samples, len(spans)):
                        b = spans[sampled]
                        yield a, b
                        sampled += 1


                    sampled = 0
                    select_querys = sample(querys*num_title_samples, num_title_samples)
                    while sampled < num_title_samples:
                        a=select_querys[sampled]+ " || title"
                        if mark_pretraining:
                            a += " || p"
                        b = title.strip() + " " + delimiter
                        yield a, b
                        sampled += 1


                    if len(querys)>1:
                        sampled = 0
                        select_querys = sample(querys*num_query_samples, num_query_samples)
                        while sampled < num_query_samples:
                            a = select_querys[sampled]+ " || query"
                            if mark_pretraining:
                                a += " || p"
                            num=0
                            while select_querys[sampled]==querys[num]:
                                num+=1
                                if(num>=len(querys)):
                                    num-=1
                                    break
                            b = querys[num].strip()
                            yield a, b
                            sampled += 1
                except Exception as e:
                    print(e)
                    continue
    else:
        raise ValueError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--delim', default="@@")
    parser.add_argument('--format', choices=['kilt', 'dpr'], default='dpr')
    parser.add_argument('--min_length_input', type=int, default=10)
    parser.add_argument('--max_length_input', type=int, default=10)
    parser.add_argument('--min_length_output', type=int, default=10)
    parser.add_argument('--max_length_output', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_title_samples', type=int, default=3)
    parser.add_argument('--num_query_samples', type=int, default=5)
    parser.add_argument('--full_doc_n', type=int, default=1)
    parser.add_argument('--mark_pretraining', action="store_true")
    parser.add_argument('--pid2query', default=None, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.source, 'w', 2 ** 20) as src, open(args.target, 'w', 2 ** 20) as tgt:
    
        for i, (s, t) in enumerate(preprocess_file(
            args.input,
            format=args.format,
            num_samples=args.num_samples,
            num_title_samples=args.num_title_samples,
            num_query_samples=args.num_query_samples,
            full_doc_n=args.full_doc_n,
            delimiter=args.delim,
            min_length_input=args.min_length_input,
            max_length_input=args.max_length_input,
            min_length_output=args.min_length_output,
            max_length_output=args.max_length_output,
            mark_pretraining=args.mark_pretraining,
            pid2query =args.pid2query,
        )):

            if random.random() < 0.1:
                s = s.lower()

            s = " " + s
            t = " " + t

            src.write(s + '\n')
            tgt.write(t + '\n')

if __name__ == '__main__':

    main()
