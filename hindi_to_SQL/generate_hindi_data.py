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
    data_path  = "wikisql_data/wikitrain_short.jsonl"
    file = open("hindi_data/wikitrain.jsonl", "w")
    print("train:")
    i = 0
    for line in open(data_path, encoding="utf8"):
        try:
            d = json.loads(line)
            result = translator.translate(d['question'], src='en', dest ='hi')
            d['question'] = translator.translate(result.text, src='hi', dest ='en').text
            # d['question'] = replace_eng(d['question'])
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

    data_path  = "wikisql_data/wikidev_short.jsonl"
    file = open("hindi_data/wikidev.jsonl", "w")
    print("dev:")
    i = 0
    for line in open(data_path, encoding="utf8"):
        try:
            d = json.loads(line)
            result = translator.translate(d['question'], src='en', dest ='hi')
            d['question'] = translator.translate(result.text, src='hi', dest ='en').text
            # d['question'] = replace_eng(d['question'])
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

    data_path  = "wikisql_data/wikitest_short.jsonl"
    file = open("hindi_data/wikitest.jsonl", "w")
    print("test:")
    i = 0
    for line in open(data_path, encoding="utf8"):
        try:
            d = json.loads(line)
            result = translator.translate(d['question'], src='en', dest ='hi')
            d['question'] = translator.translate(result.text, src='hi', dest ='en').text
            # d['question'] = replace_eng(d['question'])
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

dump_hindi()