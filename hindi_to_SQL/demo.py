# All imports
import argparse
import os
import sys
import shutil
import datetime
import torch.utils.data as torch_data
from util import *
from featurizer import *
from base_model import *
from torch_model import *
from evaluator import *
import json
from googletrans import Translator

translator = Translator()

def get_lists(q):
    ret1 = []
    ret2 = [0]
    words = q.split(" ")
    for i, word in enumerate(words):
        for j in range(len(word)+1):
            ret1.append(i)
        ret2.append(ret2[i]+len(word)+1)
    return words, ret1, ret2[:-1]


def create_json(hq, col_metadata, info):
    file = open("demo_data/wikitest.jsonl", "w")
    d = {}
    print()
    print(hq)
    d['qid'] = 0
    d['question'] = translator.translate(hq, src='hi', dest ='en').text
    print(d['question'])
    wrds, r1, r2 = get_lists(d['question'])
    d['char_to_word'] = r1
    d['word_to_char_start'] = r2
    d['tokens'] = wrds
    d['value_start_end'] = {"blank": [-1, -1]}
    d['table_id'] = "t_id"
    d['column_meta'] = col_metadata
    a = direct_agg_process(d['question'])
    s = direct_op_process(d['question'])
    info.append(a)
    info.append(s)
    d['agg'] = a
    d['select'] = s
    d['conditions'] = [[-1, -1, "blank"]]
    d['valid'] = True
    strng = json.dumps(d)[:-1]+", \"hindi_question\": \""+hq+"\"}\n"
    file.write(strng)
    file.close()
    return info

def project_demo(hq, tid):
    # Put correct model path and epoch
    week_rec_cols = [["Week", "real", None], ["Date", "string", None], ["Opponent", "string", None], ["Result", "string", None], ["Record", "string", None], ["Game Site", "string", None], ["Attendance", "real", None]]
    race_cols = [["Season", "string", None], ["Series", "string", None], ["Team", "string", None], ["Races", "real", None], ["Wins", "real", None], ["Poles", "real", None], ["F/Laps", "real", None], ["Podiums", "real", None], ["Points", "real", None], ["Position", "string", None]]
    party_cols = [["District", "string", None], ["Incumbent", "string", None], ["Party", "string", None], ["First elected", "real", None], ["Result", "string", None], ["Candidates", "string", None]]
    artist_cols = [["Position", "real", None], ["Artist", "string", None], ["Song title", "string", None], ["Highest position", "real", None], ["Points", "real", None]]
    tables = [week_rec_cols,
    race_cols,
    party_cols,
    artist_cols]
    col_metadata = tables[tid]
    model_path_demo = "models/hindi_model"
    epoch_demo = 4
    config_demo = read_conf("wikisql_demo.conf")
    featurizer_demo = HydraFeaturizer(config_demo)
    model_demo = create_model(config_demo, is_train=False)
    model_demo.load(model_path_demo, epoch_demo)
    info = []
    create_json(hq, col_metadata, info)
    info.append(tid)
    note = "_demo"
    evaluator_demo = HydraEvaluator(model_path_demo, config_demo, featurizer_demo, model_demo, note)
    return evaluator_demo.eval(epoch_demo, info)

