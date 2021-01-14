import pdb
import numpy as np
from scipy import spatial
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import json
import random
import os
import pickle
from utils.conceptnet import merged_relations

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

def relid_to_str(rel):
    global id2relation
    if rel < len(id2relation):
        return id2relation[rel]
    else:
        return id2relation[rel - len(id2relation)] + "*"

global concept2id, id2concept, relation2id, id2relation
cpnet_vocab_path = './data/cpnet/concept.txt'
load_resources(cpnet_vocab_path)
pruned_paths_path = './data/csqa/paths/dev.paths.pruned.jsonl'
output_path = './data/csqa/paths/dev.paths.test.jsonl'
with open(pruned_paths_path, 'r', encoding='utf-8') as fin_pf, \
    open(output_path, 'w', encoding='utf-8') as fout:
    for line_pf in tqdm(fin_pf, total=5000):
        qa_pairs = json.loads(line_pf)
        statement_paths = []
        statement_rel_list = []
        for qas in qa_pairs:
            if qas["pf_res"] is None:
                continue
            for item in qas["pf_res"]:
                item["path"] = [id2concept[idx] for idx in item["path"]]
                # try:
                item["rel"] = [[relid_to_str(idx) for idx in rel] for rel in item['rel']]
                # except:
                    # pdb.set_trace()
        fout.write(json.dumps(qa_pairs, indent=4) + '\n')
