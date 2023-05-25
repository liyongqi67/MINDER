# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import logging
import multiprocessing
import re

import ftfy
import torch
import tqdm
import pickle
from seal.index import FMIndex
from datasets import load_dataset
import json

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def process(line):
    tokens = tokenize(line)
    return tokens


def preprocess_file(input_path, labels, format="kilt", lowercase=False, tokenize=False, pid2query=None):


    if pid2query:
        with open(args.pid2query, 'rb') as f:
            pid2query = pickle.load(f)
    if args.id2code:
        with open(args.id2code, 'rb') as f:
            id2code = pickle.load(f)
    if format =="msmarco":

        f = load_dataset(input_path, split="train")    

        pieces_it = ((pp['docid'], pp["title"], pp["text"]) for pp in f)

        pieces_it = tqdm.tqdm(pieces_it)

        for idx, title, text in pieces_it:

            idx = idx.strip()
            title = title.strip()

            text = re.sub(r"\s+", " ", text)
            text = ftfy.fix_text(text)
            text = text.replace("BULLET::::", "")
            text = text.replace("SECTION::::", "")
            text = text.strip()

            if args.include_query:
                if args.query_format == 'free':
                    queries = pid2query[idx]
                    query = ''
                    for s in queries:
                        query =query + " " + s
                if args.query_format == 'stable':
                    queries = pid2query[idx]
                    query = ''
                    for s in queries:
                        query =query + " || " + s + ' @@'

            if args.include_code:
                code = id2code[idx]
                code =  " || " + ' '.join([*code.strip()]) + ' @@'
            if not text:
                continue

            if tokenize:
                title = " ".join(word_tokenize(title))
                text = " ".join(word_tokenize(text))

            title = f"{title} {args.delim}"

            if args.include_title and title:
                text = f"{title} {text}"
            if args.include_query and query:
                text = f"{text} {query}"
            if lowercase:
                text = text.lower()
            labels.append(idx)

            yield text



    else:
        with open(input_path, "r", 2**16) as f:

            if format == "dpr":
                next(f)
                pieces_it = csv.reader(f, delimiter="\t", quotechar='"')
                pieces_it = ((pp[0], pp[2], pp[1]) for pp in pieces_it if len(pp) == 3)

            elif format == "kilt":

                pieces_it = (line.strip() for line in f)
                pieces_it = (line.split("\t", 2) for line in pieces_it)
                pieces_it = ((pp[0], pp[1], pp[2]) for pp in pieces_it if len(pp) == 3)

            pieces_it = tqdm.tqdm(pieces_it)

            for idx, title, text in pieces_it:

                idx = idx.strip()
                title = title.strip()

                text = re.sub(r"\s+", " ", text)
                text = ftfy.fix_text(text)
                text = text.replace("BULLET::::", "")
                text = text.replace("SECTION::::", "")
                text = text.strip()

                if args.include_query:
                    if args.query_format == 'free':
                        queries = pid2query[idx]
                        query = ''
                        for s in queries:
                            query =query + " " + s
                    if args.query_format == 'stable':
                        queries = pid2query[idx]
                        query = ''
                        for s in queries:
                            query =query + " || " + s + ' @@'
                if args.include_code:
                    code = id2code[idx]
                    code =  " || " + ' '.join([*code.strip()]) + ' @@'

                if not text:
                    continue

                if tokenize:
                    title = " ".join(word_tokenize(title))
                    text = " ".join(word_tokenize(text))

                title = f"{title} {args.delim}"

                if args.include_title and title:
                    text = f"{title} {text}"
                if args.include_query and query:
                    text = f"{text} {query}"
                if args.include_code and code:
                    text = f"{text} {code}"
                if lowercase:
                    text = text.lower()
                labels.append(idx)

                yield text



def build_index(input_path):

    labels = []
    index = FMIndex()

    lines = preprocess_file(input_path, labels, args.format, lowercase=args.lowercase, tokenize=args.tokenize, pid2query=args.pid2query)

    print('start build index')
    with multiprocessing.Pool(args.jobs) as p:
        sequences = p.imap(process, lines)
        index.initialize(sequences)
    print('start build index 2')
    index.labels = labels

    return index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include_title", action="store_true")
    parser.add_argument("--delim", default="@@")
    parser.add_argument("--format", choices=["kilt", "dpr", "msmarco"], default="kilt")
    parser.add_argument("--hf_model", default=None, type=str)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument('--pid2query', default=None, type=str)
    parser.add_argument('--id2code', default=None, type=str)
    parser.add_argument("--include_query", action="store_true")
    parser.add_argument("--include_code", action="store_true")
    parser.add_argument("--query_format", choices=["free", "stable"], default="free")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print(args)

    if args.tokenize:
        from spacy.lang.en import English

        nlp = English()
        _tokenizer = nlp.tokenizer

        def word_tokenize(text):
            return [t.text.strip() for t in _tokenizer(text)]

    if args.hf_model is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
        is_bart = "bart" in args.hf_model

        def tokenize(text):
            text = text.strip()
            if is_bart:
                text = " " + text
            with tokenizer.as_target_tokenizer():
                return tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    else:
        bart = torch.hub.load("pytorch/fairseq", "bart.large").eval()

        def tokenize(text):
            return bart.encode(" " + text.strip()).tolist()[1:]

    delim = tokenize(args.delim)[:-1]
    index = build_index(args.input)
    print('start build index 3')
    index.save(args.output)
