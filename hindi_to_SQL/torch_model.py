# torch model

import torch
import os
import numpy as np
import transformers
from torch import nn
from base_model import *

class HydraTorch(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = HydraNet(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer, self.scheduler = None, None

    def train_on_batch(self, batch):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["learning_rate"]))
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()

        self.model.train()
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        batch_loss = torch.mean(self.model(**batch)["loss"])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return batch_loss.cpu().detach().numpy()

    def model_inference(self, model_inputs):
        self.model.eval()
        model_outputs = {}
        batch_size = 64 # 512
        for start_idx in range(0, model_inputs["input_ids"].shape[0], batch_size):
            input_tensor = {k: torch.from_numpy(model_inputs[k][start_idx:start_idx+batch_size]).to(self.device) for k in ["input_ids", "input_mask", "segment_ids", "input_ids_eng", "input_mask_eng", "segment_ids_eng"]}
            with torch.no_grad():
                model_output = self.model(**input_tensor)
            for k, out_tensor in model_output.items():
                if out_tensor is None:
                    continue
                if k not in model_outputs:
                    model_outputs[k] = []
                model_outputs[k].append(out_tensor.cpu().detach().numpy())

        for k in model_outputs:
            model_outputs[k] = np.concatenate(model_outputs[k], 0)

        return model_outputs

    def save(self, model_path, epoch):
        if "SAVE" in self.config and "DEBUG" not in self.config:
            save_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            print("Model saved in path: %s" % save_path)

    def load(self, model_path, epoch):
        pt_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
        loaded_dict = torch.load(pt_path, map_location=torch.device(self.device))
        if torch.cuda.device_count() > 1:
            self.model.module.load_state_dict(loaded_dict)
        else:
            self.model.load_state_dict(loaded_dict)
        print("PyTorch model loaded from {0}".format(pt_path))

class HydraNet(nn.Module):
    def __init__(self, config):
        super(HydraNet, self).__init__()
        self.config = config
        self.base_model = create_base_model(config)
        self.bert_model = create_bert_model()
        

        # #=====Hack for RoBERTa model====
        # self.base_model.config.type_vocab_size = 2
        # single_emb = self.base_model.embeddings.token_type_embeddings
        # self.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        # self.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]), requires_grad=True)
        # #====================================

        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)

        bert_hid_size = self.base_model.config.hidden_size
        projected_size = 64

        bert_english_hid_size = self.bert_model.config.hidden_size
        
        self.projection = nn.Linear(bert_hid_size, int(projected_size/4))
        self.projection_english = nn.Linear(bert_english_hid_size, int(projected_size*3/4))
        self.column_func = nn.Linear(projected_size, 3)
        self.agg = nn.Linear(projected_size, int(config["agg_num"]))
        self.op = nn.Linear(projected_size, int(config["op_num"]))
        self.where_num = nn.Linear(projected_size, int(config["where_column_num"]) + 1)
        self.start_end = nn.Linear(projected_size, 2)

        # self.column_func = nn.Linear(bert_hid_size, 3)
        # self.agg = nn.Linear(bert_hid_size, int(config["agg_num"]))
        # self.op = nn.Linear(bert_hid_size, int(config["op_num"]))
        # self.where_num = nn.Linear(bert_hid_size, int(config["where_column_num"]) + 1)
        # self.start_end = nn.Linear(bert_hid_size, 2)

    def forward(self, input_ids, input_mask, segment_ids, input_ids_eng, input_mask_eng, segment_ids_eng, agg=None, select=None, where=None, where_num=None, op=None, value_start=None, value_end=None):
        # print("[inner] input_ids size:", input_ids.size())
        if self.config["base_class"] == "roberta":
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                return_dict=False)
        elif self.config["base_class"] == "muril":
            bert_output, pooled_output, _ = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)
            bert_english, pooled_english = self.bert_model(
                input_ids=input_ids_eng,
                attention_mask=input_mask_eng,
                token_type_ids=segment_ids_eng,
                return_dict=False)
        else:
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)
        # print("pooled output before: ")
        # print(pooled_output.shape)

        # print("bert output before: ")
        # print(bert_output.shape)

        bert_output = self.projection(bert_output)
        pooled_output = self.projection(pooled_output)

        bert_english = self.projection_english(bert_english)
        pooled_english = self.projection_english(pooled_english)
        
        # print("pehgle")
        # print("pooled output: ")
        # print(pooled_output.shape)

        # print("bert output: ")
        # print(bert_output.shape)

        bert_output = self.dropout(bert_output)
        pooled_output = self.dropout(pooled_output)

        bert_output = torch.cat((bert_english, bert_output), -1)
        pooled_output = torch.cat((pooled_english, pooled_output), -1)

        # print("baad")
        # print("pooled output: ")
        # print(pooled_output.shape)

        # print("bert output: ")
        # print(bert_output.shape)

        column_func_logit = self.column_func(pooled_output)
        agg_logit = self.agg(pooled_output)
        op_logit = self.op(pooled_output)
        where_num_logit = self.where_num(pooled_output)
        start_end_logit = self.start_end(bert_output)
        value_span_mask = input_mask_eng.to(dtype=bert_output.dtype)
        # print("shape:: vsm: ", value_span_mask.shape)
        # print("shape:: sel: ", start_end_logit[:, :, 0].shape)
        # value_span_mask[:, 0] = 1
        start_logit = start_end_logit[:, :, 0] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = start_end_logit[:, :, 1] * value_span_mask - 1000000.0 * (1 - value_span_mask)

        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            loss = cross_entropy(agg_logit, agg) * select.float()
            loss += bceloss(column_func_logit[:, 0], select.float())
            loss += bceloss(column_func_logit[:, 1], where.float())
            loss += bceloss(column_func_logit[:, 2], (1-select.float()) * (1-where.float()))
            loss += cross_entropy(where_num_logit, where_num)
            loss += cross_entropy(op_logit, op) * where.float()
            loss += cross_entropy(start_logit, value_start)
            loss += cross_entropy(end_logit, value_end)


        # return loss, column_func_logit, agg_logit, op_logit, where_num_logit, start_logit, end_logit
        log_sigmoid = nn.LogSigmoid()

        return {"column_func": log_sigmoid(column_func_logit),
                "agg": agg_logit.log_softmax(1),
                "op": op_logit.log_softmax(1),
                "where_num": where_num_logit.log_softmax(1),
                "value_start": start_logit.log_softmax(1),
                "value_end": end_logit.log_softmax(1),
                "loss": loss}

def create_model(config, is_train = False) -> BaseModel:
    if config["model_type"] == "pytorch":
        return HydraTorch(config)
    # elif config["model_type"] == "tf":
    #     return HydraTensorFlow(config, is_train, num_gpu)
    else:
        raise NotImplementedError("model type {0} is not supported".format(config["model_type"]))