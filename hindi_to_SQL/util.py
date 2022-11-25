
# helper for question to sql metadata data

import json
import os
import string
import unicodedata
import transformers

pretrained_weights = {
    ("bert", "base"): "bert-base-uncased",
    ("bert", "large"): "bert-large-uncased-whole-word-masking",
    ("roberta", "base"): "roberta-base",
    ("roberta", "large"): "roberta-large",
    ("albert", "xlarge"): "albert-xlarge-v2"
}

def create_base_model(config):
    if config["base_class"] != "muril":
        weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertModel.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaModel.from_pretrained(weights_name)
    elif config["base_class"] == "albert":
        return transformers.AlbertModel.from_pretrained(weights_name)
    elif config["base_class"] == "muril":
        return transformers.AutoModel.from_pretrained('google/muril-base-cased',output_hidden_states=True)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))



def read_conf(conf_path):
    config = {}
    for line in open(conf_path, encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]
    config["train_data_path"] =  os.path.abspath(config["train_data_path"])
    config["dev_data_path"] =  os.path.abspath(config["dev_data_path"])

    return config

def read_jsonl(jsonl):
    for line in open(jsonl, encoding="utf8"):
        sample = json.loads(line.rstrip())
        yield sample

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\n" or c == "\r":
        return True
    cat = unicodedata.category(c)
    if cat == "Zs":
        return True
    return False

def is_punctuation(c):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(c)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(c)
  if cat.startswith("P") or cat.startswith("S"):
    return True
  return False

def basic_tokenize(doc):
    doc_tokens = []
    char_to_word = []
    word_to_char_start = []
    prev_is_whitespace = True
    prev_is_punc = False
    prev_is_num = False
    for pos, c in enumerate(doc):
        if is_whitespace(c):
            prev_is_whitespace = True
            prev_is_punc = False
        else:
            if prev_is_whitespace or is_punctuation(c) or prev_is_punc or (prev_is_num and not str(c).isnumeric()):
                doc_tokens.append(c)
                word_to_char_start.append(pos)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            prev_is_punc = is_punctuation(c)
            prev_is_num = str(c).isnumeric()
        char_to_word.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word, word_to_char_start
