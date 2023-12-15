# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import jsonlines
import csv
import ast
import pathlib

import json
from enum import unique, Enum

from pyserini.query_iterator import QueryIterator, DefaultQueryIterator, KiltQueryIterator
from pyserini.output_writer import OutputWriter, TrecWriter, MsMarcoWriter

from seal.retrieval import SEALDocument
from datasets import load_dataset
@unique
class TopicsFormat(Enum):
    DEFAULT = 'default'
    KILT = 'kilt'
    KILT_TEMPLATE = 'kilt_template'
    DPR = 'dpr'
    DPR_QAS = 'dpr_qas'
    DPR_QAS_TRAIN = 'dpr_qas_train'
    NQ = 'nq'
    MSMARCO = 'msmarco'
    MSMARCO_TRAIN = 'msmarco_train'
@unique
class OutputFormat(Enum):
    TREC = 'trec'
    MSMARCO = 'msmarco'
    KILT = 'kilt'
    DPR = 'dpr'

class MsmarcoQueryIterator(QueryIterator):
    def get_query(self, id_):
        return self.topics[id_]['query']

    @classmethod
    def from_topics(cls, topics_path: str):
        topics = {}
        order = []
        data = load_dataset(topics_path, split="dev")
        num = 0
        for instance in data:
            topics[int(instance['query_id'])] = instance
            order.append(int(instance['query_id']))
            num+=1

            # if num>10:
            #     break
        return cls(topics, order)
class MsmarcoQueryIteratorTrain(QueryIterator):
    def get_query(self, id_):
        return self.topics[id_]['query']

    @classmethod
    def from_topics(cls, topics_path: str):
        topics = {}
        order = []
        path = topics_path.split('_')[0]
        chunk_num = int(topics_path.split('_')[1])
        data = load_dataset(path, split="train")
        num = 1
        for instance in data:
            if num>(chunk_num-1)*10000 and num<=chunk_num*10000:
                topics[int(instance['query_id'])] = instance
                order.append(int(instance['query_id']))
            num+=1
        return cls(topics, order)
class DprQueryIterator(QueryIterator):

    def get_query(self, id_):
        return self.topics[id_]['question']

    @classmethod
    def from_topics(cls, topics_path: str):
        topics = {}
        order = []
        with open(topics_path) as fin:
            for id_, instance in enumerate(json.load(fin)):
                topics[id_] = instance
                order.append(id_)
        return cls(topics, order)


class DprQueryQasIteratorTrain(QueryIterator):

    def get_query(self, id_):
        return self.topics[id_]['question']

    @classmethod
    def from_topics(cls, topics_path: str):
        topics = {}
        order = []
        num = 0 
        order = []
        with open(topics_path) as fin:
            
            for id_, instance in enumerate(json.load(fin)[50000:60000]):
                # if num <10000:
                    topics[id_] = {
                            "question": instance['question'],
                            "answers": instance['answers'],
                        }
                    order.append(id_)
                # num+=1
        return cls(topics, order)


class DprQueryQasIterator(QueryIterator):

    def get_query(self, id_):
        return self.topics[id_]['question']

    @classmethod
    def from_topics(cls, topics_path: str):
        topics = {}
        order = []
        num = 0 
        with open(topics_path) as fin:
            fin = csv.reader(fin, delimiter="\t", quotechar='"')
            for id_, (query, answers) in enumerate(fin):
                # if num <10:
                    answers = ast.literal_eval(answers)
                    assert isinstance(answers, list) and isinstance(answers[0], str)
                    topics[id_] = {
                        "question": query,
                        "answers": answers,
                    }
                    order.append(id_)
                # num+=1
        return cls(topics, order)

class KiltTemplateQueryIterator(KiltQueryIterator):

    def get_query(self, id_):
        return self.topics[id_]['meta']['template_questions'][0]

class NqQueryIterator(QueryIterator):

    def get_query(self, id_):
        return self.topics[id_]['question_text']

    @classmethod
    def from_topics(cls, topics_path: str):
        topics = {}
        order = []

        with jsonlines.open(topics_path) as fin:
            for instance in fin:
                topics[instance['example_id']] = instance

        return cls(topics, order)

    
def get_query_iterator(topics_path: str, topics_format: TopicsFormat, queries_path: Optional[str] = None):
    mapping = {
        TopicsFormat.DEFAULT: DefaultQueryIterator,
        TopicsFormat.KILT: KiltQueryIterator,
        TopicsFormat.KILT_TEMPLATE: KiltTemplateQueryIterator,
        TopicsFormat.DPR: DprQueryIterator,
        TopicsFormat.DPR_QAS: DprQueryQasIterator,
        TopicsFormat.DPR_QAS_TRAIN: DprQueryQasIteratorTrain,
        TopicsFormat.NQ: NqQueryIterator,
        TopicsFormat.MSMARCO: MsmarcoQueryIterator,
        TopicsFormat.MSMARCO_TRAIN: MsmarcoQueryIteratorTrain,
    }
    return mapping[topics_format].from_topics(topics_path)

class KiltWriter(OutputWriter):

    def write(self, topic: str, hits: list):
        provenance = []
        datapoint = {'id': topic, 'input': None, 'output': [{'provenance': provenance}]}
        for docid, rank, score, hit in self.hits_iterator(hits):
            if isinstance(hit, SEALDocument):
                if datapoint['input'] is None and hit.query is not None:
                    datapoint['input'] = hit.query
                docid = docid.split("-")
                wikipedia_id = int(docid[0])
                start_paragraph_id = end_paragraph_id = 0
                if len(docid) == 2:
                    start_paragraph_id = end_paragraph_id = int(docid[1])
                elif len(docid) >= 3:
                    start_paragraph_id = int(docid[1])
                    end_paragraph_id = int(docid[2])
                title, body = hit.text()
                provenance.append({
                    "wikipedia_id": wikipedia_id, 
                    "start_paragraph_id": start_paragraph_id,  
                    "end_paragraph_id": end_paragraph_id, 
                    "text": f'{title} @@ {body}',
                    "score": score,
                })
                if hit.keys is not None:
                    provenance[-1]['meta'] = {'keys': hit.keys}
            else:
                provenance.append({"wikipedia_id": docid})
        json.dump(datapoint, self._file)
        self._file.write('\n')

class DprWriter(OutputWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = []

    def write(self, topic: str, hits: list):
        datapoint = self.topics[topic]
        self.order.append(topic)
        ctxs = datapoint['ctxs'] = []
        for docid, rank, score, hit in self.hits_iterator(hits):
            title, body = hit.text()
            ctx = {
                "title": title.strip(),
                "text": body.strip(),
                "score": score,
                "passage_id": docid, 
                "keys":hit.keys
            }
            ctxs.append(ctx)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        data = [self.topics[t] for t in self.order]
        json.dump(data, self._file, indent="    ")
        return super().__exit__(exc_type, exc_value, exc_traceback) 

def get_output_writer(file_path: str, output_format: OutputFormat, *args, **kwargs) -> OutputWriter:
    mapping = {
        OutputFormat.TREC: TrecWriter,
        OutputFormat.MSMARCO: MsMarcoWriter,
        OutputFormat.KILT: KiltWriter,
        OutputFormat.DPR: DprWriter,
    }
    return mapping[output_format](file_path, *args, **kwargs)
