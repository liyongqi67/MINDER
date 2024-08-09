import torch
import json
import os
import time
import threading
from tqdm import tqdm
from more_itertools import chunked
import argparse
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def setup_ddp():
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

def get_teacher_batch_score(args, model, tokenizer, input, device):
    
    batch_dict = tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    score = (embeddings[:1] @ embeddings[1:].T) * 100
    score = score.tolist()
    return score

'''
def process_file(args, in_file, gpus, pbar):
    out_file = os.path.join(args.teacher_output_path, os.path.basename(in_file))
    with open(in_file, 'r', encoding='UTF-8') as top_200:
        data = [json.loads(line) for line in top_200]
        # data = []
        # for i in range(20):
        #     data.append(json.loads(top_200.readline()))
    print("---data_loaded for file: {} -----".format(in_file))
    file_chunk_size = int(len(data) / len(gpus))
    if len(data) % len(gpus) > 0:
        file_chunk_size += 1

    file_chunks = list(chunked(data, file_chunk_size))
    threads = []
    for idx, data_chunk in enumerate(file_chunks):
        device = gpus[idx % len(gpus)]
        print("__GPU: start gen: ", idx)
        targetFn=gen_teacher_scores_with_file_chunk_E5
        if args.teacher_name == "simLM_msmarco" or args.teacher_name == "simLM_NQ":
            print(args.teacher_name)
            targetFn=gen_teacher_scores_with_file_chunk_simLM
        thread = threading.Thread(target=targetFn, args=(args, device, data_chunk, pbar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    with open(out_file, 'w') as f:
        idx = 0
        for data_line in data:
            line = json.dumps(data_line)
            if idx > 0:
                f.write('\n')
            f.write(line)
            idx += 1
        print('__SUCC: ', idx)


def get_teacher_scores_with_multi_gpu(args):
    gpus = args.device.split(",")
    originial_path = args.original_path
    files_to_process = [f for f in os.listdir(originial_path) if os.path.isfile(os.path.join(originial_path, f))]

    pbar = tqdm(total=len(files_to_process))

    for filename in files_to_process:
        in_file = os.path.join(originial_path, filename)
        process_file(args, in_file, gpus, pbar)
        pbar.update()

    pbar.close()

'''

def load_mamarco_testdata(path):
    data = {}
    with open(path, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid = int(l[0])
            pid = int(l[1])
            if qid not in data:
                data[qid] = []
            data[qid].append(pid)
        print(len(data))
    res = []
    for k in data:
        res.append({k: data[k]})
        

    return res

def get_teacher_scores_with_multi_gpu(args):
    gpus = args.device.split(",")
    originial_path, teacher_output_path = args.original_path, args.teacher_output_path
 
    print("__GPU: start with multi-gpu: ", gpus)

    in_file = originial_path + '/' + args.original_file
    out_file = teacher_output_path + '/' + args.original_file
    print("in_file:", in_file)
    print("---data_start_process-----")
    with open(in_file, 'r', encoding = 'UTF-8') as top_200:
        if args.do_train:
            if args.if_MSMarco_devdata:
                data = load_mamarco_testdata('./evaluation_test_output/output_test_msmarco_minder.json')
                #data = load_mamarco_testdata('./evaluation_test_output/test_msmarco.json')
                print("---mamarco_testdata data_loaded-----")
                print(len(data))
            else:
                data = [json.loads(line) for line in top_200]
                print("---train data_loaded-----")
        # data = []
        # for i in range(20):
        #     data.append(json.loads(top_200.readline()))
        else:
            data = json.load(top_200)
            print("---test data_loaded-----")


    file_chunk_size = int(len(data) / len(gpus))
    if len(data) % len(gpus) > 0:
        file_chunk_size += 1
   
    file_chunk = list(chunked(data, file_chunk_size))
    
    pbar = tqdm(total=len(data))
    file_chunk_num = len(file_chunk)

    print("__GEN: chunk size:", file_chunk_size, file_chunk_num, in_file)
    
    pool = []
    scores = []
    for idx in range(file_chunk_num):
        device = gpus[idx]
        data_chunk = file_chunk[idx]

        print("__GPU: start gen: ", idx, file_chunk_num)
        if args.do_train:
            if args.teacher_name == 'E5':
                targetFn=gen_teacher_scores_with_file_chunk_E5
                print(args.teacher_name)
            elif args.teacher_name == "simLM" or args.teacher_name == "simLM_NQ" or args.teacher_name == 'simLM_msmarco':
                print(args.teacher_name)
                targetFn=gen_teacher_scores_with_file_chunk_simLM
        elif args.do_test:
            print('do_test')
            targetFn = get_teacher_scores_for_testComnined_multigpu
        t = threading.Thread(target=targetFn, args=(args, device, data_chunk, pbar))
        pool.append(t)

    
    # start all
    for p in pool:
        p.start()
    
    # join all
    for p in pool:
        p.join()
    if args.do_train:
        with open(out_file, 'w') as f:
            idx = 0
            for data_line in data:
                line = json.dumps(data_line)
                if idx > 0:
                    f.write('\n')
                f.write(line)
                idx += 1
            print('__SUCC: ', idx)
    else:
        output_path = teacher_output_path + '/' + args.teacher_kind  + args.original_file
        with open(output_path, 'w') as f:
            test_json = json.dumps(data)
            f.write(test_json)


# simLM batch encode
def batch_encode_simLM(tokenizer, queries, passages, titles, max_length=192, batch_size=20, device=None):
    # encoded_batches = {}

        encoded_batch = {
                'input_ids': [],
                'attention_mask': [],
                'token_type_ids': []
            }
        for i in range(0, len(passages), batch_size):
            print(i)
            
            batch_queries = queries * min(batch_size, len(passages) - i)
            batch_passages = passages[i:i + batch_size]
            batch_titles = titles[i:i + batch_size]
            # print(len(batch_titles[0]))
            # print(len(batch_passages[0]))

            text_pairs = [f'{t}: {p}' if t and p else '' for t, p in zip(batch_titles, batch_passages)]

            # batch_queries = torch.tensor(batch_queries).to(device)
            # text_pair = torch.tensor(text_pair).to(device)
           
            #print(text_pairs)
            encoded = tokenizer(batch_queries, text_pairs, max_length=192, padding="max_length", truncation=True, return_tensors='pt')
            len_input_ids  = len(encoded['input_ids'])
            # if len_input_ids  != 192:
            #     pad_extra = [0] *(192- len_input_ids )
            encoded_batch['input_ids'].extend(encoded['input_ids'].tolist())                    
            encoded_batch['attention_mask'].extend(encoded['attention_mask'].tolist())            
            encoded_batch['token_type_ids'].extend(encoded['token_type_ids'].tolist())   
        encoded_batch['input_ids'] = torch.tensor(encoded_batch['input_ids'])
        encoded_batch['attention_mask'] = torch.tensor(encoded_batch['attention_mask'])
        encoded_batch['token_type_ids'] = torch.tensor(encoded_batch['token_type_ids'])
        encoded_batch = {key: torch.tensor(value).to(device) for key, value in encoded_batch.items()}
        # print(encoded_batch)
        #print("encoded_batch:", encoded_batch["input_ids"].size())

        return encoded_batch



# simLM batch inference
def batch_inference_simLM(model, encoded_batch):
    with torch.no_grad():
        outputs = model(**encoded_batch, return_dict=True)
        # print(outputs.logits.size())
        # print(outputs.logits)
        # print(outputs.logit.size())

        # prob = F.softmax(outputs.logits, dim=0)

        # print(prob.size())
        #print(outputs)
        # print("-----")
        # print(len(outputs), len(outputs[0]), len(outputs[0][0]))
    # return prob
    return outputs.logits


def gen_teacher_scores_with_file_chunk_simLM(args, device, data, pbar):
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path).to(device)
    model.eval()
    # print(tokenizer)
    name = "score_teacher"+'_' + args.teacher_name
    #print(args.teacher_name, args.teacher_kind)
    for idx in range(len(data)):

        try:
            line = data[idx]
            #print(line)
            if args.if_MSMarco_devdata:
                queries_list = [args.qid2query[list(line.keys())[0]]]
                #print(type(queries_list))
                
            else:
                queries_list = [line["question"]]
            if args.if_MSMarco:
                if args.if_MSMarco_devdata:
                    positive_passages_list = [args.dataset_msmarco[pid]['text'] for pid in list(line.values())[0]]
                    #print('len passage:', len(positive_passages_list))
                    negative_passages_list = []
                    positive_titles = [args.dataset_msmarco[pid]['title'] for pid in list(line.values())[0]]
                    negative_titles = []
                else:
                    #print(len(args.dataset_msmarco))
                    positive_passages_list = [args.dataset_msmarco[int(passage['passage_id'])]['text'] for passage in line["positive_ctxs"]]
                    negative_passages_list = [args.dataset_msmarco[int(passage['passage_id'])]['text'] for passage in line["negative_ctxs"]]
                    #print('len(negative_passages_list: ',len(negative_passages_list))
            else:
                positive_passages_list = [passage["text"] for passage in line["positive_ctxs"]]
                negative_passages_list = [passage["text"] for passage in line["negative_ctxs"]]
            if not args.if_MSMarco_devdata:
                positive_titles = [passage_["title"] for passage_ in line["positive_ctxs"]]
                negative_titles = [passage_["title"] for passage_ in line["negative_ctxs"]]

            
            batch_queries = queries_list
            batch_title = positive_titles + negative_titles
            batch_passages = positive_passages_list + negative_passages_list
        
            #print(len(batch_title), len(batch_passages), sep = "-----")
            # print('batch_queries', batch_queries)
            # print('batch_title',batch_title)
            # print(batch_passages)

            encoded_batch = batch_encode_simLM(tokenizer, batch_queries, batch_passages, titles = batch_title, device=device)

            logits= batch_inference_simLM(model, encoded_batch)
            

            if not args.if_MSMarco_devdata:
                lp,ln = len(line["positive_ctxs"])-1,len(line["negative_ctxs"])

                for k in range(lp+ln+1):
                    #print('k, l_p:',k, lp)
                    if lp < 0:                
                        line["negative_ctxs"][k][name] = logits[k][0].item()
                    elif k <= lp:
                        line["positive_ctxs"][k][name] = logits[k][0].item()
                    else:
                        line["negative_ctxs"][k-lp-1][name] = logits[k][0].item()
            else:
                l = len(list(line.values())[0])
                #print(l)
                line['simLM_msmarco_score'] = []
                for k in range(l):
                    line['simLM_msmarco_score'].append(logits[k][0].item())
                    #print(line)

            pbar.update(1)
        except ValueError:
            error_file = "/minder_ltr/error_file1.json"
            with open(error_file, "a+") as ef:
                ef.write(json.dumps(line)+'\n')
                
        



def gen_teacher_scores_with_file_chunk_E5(args, device, data, pbar):
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    model = AutoModel.from_pretrained(args.teacher_path).to(device)

    for idx in range(len(data)):
        line = data[idx]

        input_texts = []
        query = 'query: ' + line["question"]
        # input_texts.append(query)
        #print('lenposi, lennega:',len(line["positive_ctxs"]), len(line["negative_ctxs"]) )
        for i in range(len(line["positive_ctxs"])):
            if args.if_MSMarco:
                #print('i---:', i)
                pp_id = line["positive_ctxs"][i]['passage_id']
                text = args.dataset_msmarco[int(pp_id)]['text']
            else:
                text = line["positive_ctxs"][i]["text"]
            text = 'passage: ' + text
            input_texts.append(text)
        
        for j in range(len(line["negative_ctxs"])):
            if args.if_MSMarco:
                #print('j---:', j)
                pn_id = line["negative_ctxs"][j]['passage_id']
                text = args.dataset_msmarco[int(pn_id)]['text']
            else:
                text = line["negative_ctxs"][j]["text"]
            text = 'passage: ' + text
            input_texts.append(text)
        input_texts = list(chunked(input_texts, 20)) 
        inputs = [[query] + input_text for input_text in input_texts]
        # print(inputs)
        scores = []
        for input in inputs:
            # print(input)
            # start_time = time.time()                
            score = get_teacher_batch_score(args, model, tokenizer, input, device)
            # print("--- %s seconds ---" % (time.time() - start_time))
            scores.extend(score[0])

        l_p = len(line["positive_ctxs"])-1
        name = "score_teacher"+'_' + args.teacher_name
        # print(file_name, num_line,len(line["positive_ctxs"]), len(line["negative_ctxs"]), sep = '\n')
        for k in range(len(line["positive_ctxs"])+ len(line["negative_ctxs"])):
            #print('k, l_p:',k, l_p)
            if l_p < 0:                
                line["negative_ctxs"][k][name] = scores[k]
            elif k <= l_p:
                line["positive_ctxs"][k][name] = scores[k]
            else:
                line["negative_ctxs"][k-i-1][name] = scores[k]
        
        pbar.update(1)



# single gpu
def get_teacher_scores(args, model, tokenizer):

    model = model.to(args.device)
    originial_path, teacher_output_path = args.original_path, args.teacher_output_path
    files = os.listdir(originial_path)
    print(files)
    # json_teacher = open(path_new, 'w')
    # line_num = 0
    for file_name in files:
        if file_name != 'SEAL_ON_TRAIN1.json':
            continue
        print("__RUN Start to handle: ", file_name)
        in_file = originial_path + '/' + file_name
        out_file = teacher_output_path + '/' + file_name
        with open(in_file, 'r', encoding = 'UTF-8') as top_200:
            data = [json.loads(line) for line in top_200]
        num_line = 0
        print("__RUN Start to Infer: ", file_name)

        for idx in tqdm(range(len(data))):
            line = data[idx]

            input_texts = []
            query = 'query: ' + line["question"]
            # input_texts.append(query)
            for i in range(len(line["positive_ctxs"])):
                text = line["positive_ctxs"][i]["text"]
                text = 'passage: ' + text
                input_texts.append(text)
            for j in range(len(line["negative_ctxs"])):
                text = line["negative_ctxs"][j]["text"]
                text = 'passage: ' + text
                input_texts.append(text)
            input_texts = list(chunked(input_texts, 30)) 
            inputs = [[query] + input_text for input_text in input_texts]
            # print(inputs)
            scores = []
            for input in inputs:
                # print(input)
                # start_time = time.time()                
                score = get_teacher_batch_score(args, model, tokenizer, input)
                # print("--- %s seconds ---" % (time.time() - start_time))
                scores.extend(score[0])
            # print(scores)
            num_line += 1
            l_p = len(line["positive_ctxs"])-1
            name = "score_teacher"+'_' + args.teacher_name
            # print(file_name, num_line,len(line["positive_ctxs"]), len(line["negative_ctxs"]), sep = '\n')
            for k in range(len(line["positive_ctxs"])+ len(line["negative_ctxs"])):
                if k <= l_p:
                    line["positive_ctxs"][k][name] = scores[k]
                else:
                    line["negative_ctxs"][k-i-1][name] = scores[k]
            # torch.cuda.empty_cache()
        
        data_json = [json.dumps(data_line) for data_line in data]

        with open(out_file, 'w') as f:
            f.writelines(data_json)

# get tacher score for test data 
'''
input: output of minder generation key file
output: test data which pasaages of query sorted by teacher score, then fed with output_test.py
CUDA_VISIBLE_DEVICES=2 python check_top200.py --do_test True --device cuda:0 --teacher_kind score_teacher_simLM --teacher_name simLM 
--test_input_path /minder_ltr/evaluation --teacher_path intfloat/simlm-msmarco-reranker
'''
def get_teacher_scores_for_testComnined(args, model, tokenizer):
    model = model.to(args.device)
    test_input_path, teacher_output_path_for_test = args.test_input_path, args.test_output_path
    teacher_kind = args.teacher_kind
    teacher_name = args.teacher_name
    files = os.listdir(test_input_path)
    # print(files)
    # json_teacher = open(path_new, 'w')
    # line_num = 0
    for file_name in files:
        if file_name != 'output_test_trivia_minder.json':
        #if file_name != 'test100_trivia.json':
            continue
        print("__RUN Start to handle: ", file_name)
        in_file = test_input_path + '/' + file_name
        out_file = teacher_output_path_for_test + '/' + file_name
        with open(in_file, 'r', encoding = 'UTF-8') as f:
            data = json.load(f) 
            #data = [json.loads(line) for line in f]
        num_line = 0
        print("__RUN Start to Infer: ", file_name)
        #print(teacher_kind, teacher_name)
        #data = data[:10]
        for idx in tqdm(range(len(data))):
            line = data[idx]
            # with open("test_line.json", "w") as tf:
            #     test_json = json.dumps(line)
            #     tf.write(test_json)
            # return 
            scores = []
            if teacher_kind == "score_teacher_E5" and teacher_name == "E5":          
                input_texts = []
                for i in range(len(line["ctxs"])):
                    text = line["ctxs"][i]["text"]
                    text = 'passage: ' + text
                    input_texts.append(text) 

                query = 'query: ' + line["question"]
                input_texts = list(chunked(input_texts, 20)) 
                inputs = [[query] + input_text for input_text in input_texts]
                # print(input
                for input in inputs:
                    # print(input)
                    # start_time = time.time()                
                    score = get_teacher_batch_score(args, model, tokenizer, input, args.device)
                    # print("--- %s seconds ---" % (time.time() - start_time))
                    scores.extend(score[0])
            elif teacher_kind == "score_teacher_simLM_msmarco" or teacher_kind == "score_teacher_simLM_NQ":
            
                batch_queries = [line["question"]]
                batch_passages = [passage["text"] for passage in line["ctxs"]]
                batch_title = [passage["title"] for passage in line["ctxs"]]
               
                encoded_batch = batch_encode_simLM(tokenizer, batch_queries, batch_passages, titles = batch_title, device=args.device)
                logits= batch_inference_simLM(model, encoded_batch)
                # print(encoded_batch["input_ids"].size())
                # print(len(logits[0]))
                
                scores = [logit[0].item() for logit in logits]
                # print(scores)
                
                name = "score_teacher"+'_' + teacher_name
                #print(name)
                for k in range(len(line["ctxs"])):
                    line["ctxs"][k][name] = scores[k]
                line["ctxs"] =sorted(line["ctxs"], key = lambda x: x[teacher_kind], reverse = True)
                
        output_path = teacher_output_path_for_test + '/' + args.teacher_kind  + file_name
        with open(output_path, 'w') as f:
            test_json = json.dumps(data)
            f.write(test_json)
            
        
  
def get_teacher_scores_for_testComnined_multigpu(args, device, data, pbar):

    teacher_kind = args.teacher_kind
    teacher_name = args.teacher_name

    print(teacher_kind, teacher_name)
    if args.teacher_kind == "score_teacher_E5":
        print(1)
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
        model = AutoModel.from_pretrained(args.teacher_path)
        
    elif args.teacher_kind == "score_teacher_simLM_msmarco" or args.teacher_kind == "score_teacher_simLM_NQ":
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path)
    
    for idx in range(len(data)):
        line = data[idx]
        # with open("test_line.json", "w") as tf:
        #     test_json = json.dumps(line)
        #     tf.write(test_json)
        # return 
        scores = []
        if teacher_kind == "score_teacher_E5" and teacher_name == "E5":          
            
            
            input_texts = []
            for i in range(len(line["ctxs"])):
                text = line["ctxs"][i]["text"]
                text = 'passage: ' + text
                input_texts.append(text) 

            query = 'query: ' + line["question"]
            input_texts = list(chunked(input_texts, 20)) 
            inputs = [[query] + input_text for input_text in input_texts]
            
            for input in inputs:
                # print(input)
                # start_time = time.time()                
                score = get_teacher_batch_score(args, model, tokenizer, input, device)
                # print("--- %s seconds ---" % (time.time() - start_time))
                print(score[0])
                scores.extend(score[0])
            
        elif teacher_kind == "score_teacher_simLM_msmarco" or teacher_kind == "score_teacher_simLM_NQ":
        
            batch_queries = [line["question"]]
            batch_passages = [passage["text"] for passage in line["ctxs"]]
            batch_title = [passage["title"] for passage in line["ctxs"]]
            
            tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
            encoded_batch = batch_encode_simLM(tokenizer, batch_queries, batch_passages, titles = batch_title, device= device)
            logits= batch_inference_simLM(model, encoded_batch)
            # print(encoded_batch["input_ids"].size())
            # print(len(logits[0]))
            
            scores = [logit[0].item() for logit in logits]
            # print(scores)
            
            name = "score_teacher"+'_' + teacher_name
            #print(name)
            for k in range(len(line["ctxs"])):
                line["ctxs"][k][name] = scores[k]
            line["ctxs"] =sorted(line["ctxs"], key = lambda x: x[teacher_kind], reverse = True)
            
            pbar.update(1)
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print(111111)
    parser.add_argument('--original_path', default =  "/minder_ltr/TrainData/MINDER_on_train_top200_transformed_all", type = str)
    parser.add_argument('--original_file', default =  "SEAL_ON_TRAIN1.json", type = str)
    parser.add_argument('--teacher_output_path', type = str)
    parser.add_argument('--teacher_name', default = 'E5')
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--teacher_path', default = 'intfloat/e5-large-v2', type = str)
    parser.add_argument("--test_input_path", default = '/minder_ltr/test_input', type = str)
    parser.add_argument("--test_output_path", default = "/minder_ltr/test_output", type = str)
    parser.add_argument("--teacher_kind", default = "score_teacher_E5")
    parser.add_argument("--do_train", default = False)
    parser.add_argument("--do_test", default = False)
    parser.add_argument("--if_MSMarco", default = False)
    parser.add_argument("--if_MSMarco_devdata", default = False)
    
    parser.add_argument("--multi_gpu_test", default = False)
    # parser.add_argument("--target_func", default = "get_teacher_scores_with_multi_gpu_E5", help ="change with teacher" )
    args = parser.parse_args()
    print(22222)
    

    print('__CUDA Available Device: ', torch.cuda.device_count())
    print('__CUDA Currence Device: ', torch.cuda.current_device())

    if args.do_train:
        # model = SentenceTransformer('intfloat/e5-large-v2')
        if args.if_MSMarco:
            args.dataset_msmarco = load_dataset("Tevatron/msmarco-passage-corpus", split="train") 
            print("---msmarco train key data loaded") 
            if args.if_MSMarco_devdata:
                # args.msmarco_dev = load_dataset("Tevatron/msmarco-passage", split="dev")
                # print("---msmarco dev key data loaded") 
                args.qid2query={}
                
                for s in load_dataset("Tevatron/msmarco-passage", split="dev"):
                    args.qid2query[int(s['query_id'])] = s['query']
                    
            # get_teacher_scores_with_multi_gpu(args, dataset_msmarco)
        
        get_teacher_scores_with_multi_gpu(args)
            

    if args.do_test:
        if args.multi_gpu_test:
            get_teacher_scores_with_multi_gpu(args)
        else:
            if args.teacher_kind == "score_teacher_E5":
                print(1)
                tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
                model = AutoModel.from_pretrained(args.teacher_path)
                
            elif args.teacher_kind == "score_teacher_simLM_msmarco" or args.teacher_kind == "score_teacher_simLM_NQ":
                tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
                model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path)
            
            print(2)
            get_teacher_scores_for_testComnined(args, model, tokenizer)




# # 使用 AutoModelForSequenceClassification 加载模型
# model = AutoModelForSequenceClassification.from_pretrained("./nq-reranker/reranker-model")

# # 使用 AutoTokenizer 加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./nq-reranker/reranker-model")

# # 设置模型为评估模式
# model.eval()


   

# dataset = load_dataset("Tevatron/msmarco-passage-corpus", split="train")        
                
