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


import multiprocessing
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
    pid, 
    text, 
    title, 
    querys,
    num_samples,
    num_title_samples,
    num_query_samples,
    format, 
    delimiter, 
    min_length_input,
    max_length_input,
    min_length_output, 
    max_length_output,
    full_doc_n,
    mark_pretraining
    ):
    
    if format == 'kilt':

        raise NotImplementedError
    elif format == 'dpr':
        try:
            text = text
            title = title


            tokens = text.split()

            for _ in range(full_doc_n):
                a = text.strip() + " || title"
                if mark_pretraining:
                    a += " || p"
                b = title.strip() + " " + delimiter
                yield a, b

            sampled = 0
            failures = 0
            while sampled < num_title_samples and failures < 10:

                if random.random() > 0.5:
                    len_a = random.randint(min_length_input, max_length_input)
                    idx_a = random.randint(0, max(0, len(tokens)-len_a))
                    a = ' '.join(tokens[idx_a:idx_a+len_a]).strip() + " || title"
                    if mark_pretraining:
                        a += " || p"
                    b = title.strip() + " " + delimiter
                
                else:

                    len_b = random.randint(min_length_output, max_length_output)
                    idx_b = random.randint(0, max(0, len(tokens)-len_b))

                    if not is_good(tokens[idx_b]):
                        failures += 1
                        continue

                    b = ' '.join(tokens[idx_b:idx_b+len_b]).strip()
                    a = title.strip() + ' || body'
                    if mark_pretraining:
                        a += " || p"
                
                yield a, b
                sampled += 1

            sampled = 0
            failures = 0
            while sampled < num_samples and failures < 10:
                len_a = random.randint(min_length_input, max_length_input)
                len_b = random.randint(min_length_output, max_length_output)
                idx_a = random.randint(0, max(0, len(tokens)-len_a))
                idx_b = random.randint(0, max(0, len(tokens)-len_b))

                if idx_a == idx_b or (not is_good(tokens[idx_b])):
                    failures += 1
                    continue

                a = ' '.join(tokens[idx_a:idx_a+len_a]).strip() + ' || body'
                if mark_pretraining:
                    a += " || p"
                b = ' '.join(tokens[idx_b:idx_b+len_b]).strip()
                yield a, b
                sampled += 1



            sampled = 0
            failures = 0
            while sampled < num_query_samples and failures < 10:

                if random.random() > 0.5:
                    len_a = random.randint(min_length_input, max_length_input)
                    idx_a = random.randint(0, max(0, len(tokens)-len_a))
                    a = ' '.join(tokens[idx_a:idx_a+len_a]).strip() + " || query"
                    if mark_pretraining:
                        a += " || p"
                    b = sample(querys, 1)[0]
                
                else:

                    len_b = random.randint(min_length_output, max_length_output)
                    idx_b = random.randint(0, max(0, len(tokens)-len_b))

                    if not is_good(tokens[idx_b]):
                        failures += 1
                        continue

                    b = ' '.join(tokens[idx_b:idx_b+len_b]).strip()
                    a = title.strip() + ' || body'
                    if mark_pretraining:
                        a += " || p"
                
                yield a, b
                sampled += 1

        except Exception as e:
            print(e)

    else:
        raise ValueError
def preprocess_file_wrapper(args):
    return list(preprocess_file(*args))

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
    parser.add_argument("--jobs", type=int, default=20)
    return parser.parse_args()


def data_read(f, pid2query):        
    for pid, text, title in tqdm.tqdm(f):
        querys = pid2query[str(pid)]
        if text == title:
            continue
        if len(querys)<2:
            continue
        yield pid, text, title, querys
def main():
    args = parse_args()
    with open(args.source, 'w', 2 ** 20) as src, open(args.target, 'w', 2 ** 20) as tgt:
        if args.pid2query:
            with open(args.pid2query, 'rb') as f:
                pid2query = pickle.load(f)
        with open(args.input, 'r', 2 ** 20) as f:
            next(f)
            f = csv.reader(f, delimiter='\t', quotechar='"')
            data = data_read(f, pid2query)



            arg_it = ((pid, text, title, querys, args.num_samples, args.num_title_samples, args.num_query_samples, args.format, args.delim, 
                args.min_length_input,args.max_length_input,
             args.min_length_output, args.max_length_output, args.full_doc_n, args.mark_pretraining ) for pid, text, title, querys in data)

            with multiprocessing.Pool(args.jobs) as pool:
                for sts in  pool.imap(preprocess_file_wrapper, arg_it):
                    for s, t in sts:
                        if random.random() < 0.1:
                            s = s.lower()
                        s = " " + s
                        t = " " + t
                        src.write(s + '\n')
                        tgt.write(t + '\n')

if __name__ == '__main__':

    main()
