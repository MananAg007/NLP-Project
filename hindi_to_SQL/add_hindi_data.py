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


def dump_hindi():
    data_path  = "data/wikitrain_short.jsonl"
    file = open("train_hindi.jsonl", "w")
    print("train:")
    i = 0
    for line in open(data_path, encoding="utf8"):
        try:
            d = json.loads(line)
            result = translator.translate(d['question'], src='en', dest ='hi')
            d['question'] = translator.translate(result.text, src='hi', dest ='en').text
            wrds, r1, r2 = get_lists(d['question'])
            d['char_to_word'] = r1
            d['word_to_char_start'] = r2
            d['tokens'] = wrds
            value = d['value_start_end']
            if len(value.keys())>1:
                # print("hetre")
                continue
            # print("hh")
            value_list = list(value.keys())[0].split(" ")
            value_start = value_list[0]
            # print(value_start)
            for ii in range(len(wrds)):
                if(wrds[ii] == value_start):
                    start_end = [ii, ii+len(value_list)]
                    break
            # print(start_end)
            d['value_start_end'] = {list(value.keys())[0]: start_end}
            # print(d['value_start_end'])
            strng = json.dumps(d)[:-1]+", \"hindi_question\": \""+result.text+"\"}\n"
            file.write(strng)
            print(i)
            i+=1
        except:
            pass
    file.close()

    data_path  = "data/wikidev_short.jsonl"
    file = open("dev_hindi.jsonl", "w")
    print("dev:")
    i = 0
    for line in open(data_path, encoding="utf8"):
        try:
            d = json.loads(line)
            result = translator.translate(d['question'], src='en', dest ='hi')
            d['question'] = translator.translate(result.text, src='hi', dest ='en').text
            wrds, r1, r2 = get_lists(d['question'])
            d['char_to_word'] = r1
            d['word_to_char_start'] = r2
            d['tokens'] = wrds
            value = d['value_start_end']
            if len(value.keys())>1:
                # print("hetre")
                continue
            # print("hh")
            value_list = list(value.keys())[0].split(" ")
            value_start = value_list[0]
            # print(value_start)
            for ii in range(len(wrds)):
                if(wrds[ii] == value_start):
                    start_end = [ii, ii+len(value_list)]
                    break
            # print(start_end)
            d['value_start_end'] = {list(value.keys())[0]: start_end}
            # print(d['value_start_end'])
            strng = json.dumps(d)[:-1]+", \"hindi_question\": \""+result.text+"\"}\n"
            file.write(strng)
            print(i)
            i+=1
        except:
            pass
    file.close()

    data_path  = "data/wikitest_short.jsonl"
    file = open("test_hindi.jsonl", "w")
    print("test:")
    i = 0
    for line in open(data_path, encoding="utf8"):
        try:
            d = json.loads(line)
            result = translator.translate(d['question'], src='en', dest ='hi')
            d['question'] = translator.translate(result.text, src='hi', dest ='en').text
            wrds, r1, r2 = get_lists(d['question'])
            d['char_to_word'] = r1
            d['word_to_char_start'] = r2
            d['tokens'] = wrds
            value = d['value_start_end']
            if len(value.keys())>1:
                # print("hetre")
                continue
            # print("hh")
            value_list = list(value.keys())[0].split(" ")
            value_start = value_list[0]
            # print(value_start)
            for ii in range(len(wrds)):
                if(wrds[ii] == value_start):
                    start_end = [ii, ii+len(value_list)]
                    break
            # print(start_end)
            d['value_start_end'] = {list(value.keys())[0]: start_end}
            # print(d['value_start_end'])
            strng = json.dumps(d)[:-1]+", \"hindi_question\": \""+result.text+"\"}\n"
            file.write(strng)
            print(i)
            i+=1
        except:
            pass
    file.close()

def create_json():
    # data_path  = "data/wikitrain.jsonl"
    # file = open("train_hindi.jsonl", "w")
    # i = 0
    # for line in open(data_path, encoding="utf8"):
    #     d = json.loads(line)
    #     result = translator.translate(d['question'], src='en', dest ='hi')
    #     strng = json.dumps(d)[:-1]+", \"hindi_question\": \""+result.text+"\"}\n"
    #     file.write(strng)
    #     print(i)
    #     i+=1
    # file.close()
    # for line in open("train_hindi.jsonl", encoding="utf8"):
    #     words = line.split(", ")
    #     hq = words[0]
    #     ti = words[1]
    #     d = {}
    #     d["qid"] = 0
    #     d["question"] = translator.translate(hq, src='en', dest ='hi')
    #     d["column_meta"] = 
    pass


dump_hindi()