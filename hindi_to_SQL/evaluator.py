import os
import numpy as np
from featurizer import *
from base_model import *
from sklearn import metrics

class HydraEvaluator():
    def __init__(self, output_path, config, hydra_featurizer: HydraFeaturizer, model:BaseModel, note=""):
        self.config = config
        self.model = model
        self.eval_history_file = os.path.join(output_path, "eval.log")
        self.bad_case_dir = os.path.join(output_path, "bad_cases"+note)
        self.note = note
        try:
            os.mkdir(output_path)
        except:
            pass
        if "DEBUG" not in config:
            try:
                os.mkdir(self.bad_case_dir)
            except:
                pass
            with open(self.eval_history_file, "w", encoding="utf8") as f:
                f.write(note.rstrip() + "\n")

        self.eval_data = {}
        if note == "":
            for eval_path in config["dev_data_path"].split("|") + config["test_data_path"].split("|"):
                if note == "":
                    eval_data = SQLDataset(eval_path, config, hydra_featurizer, True)
                self.eval_data[os.path.basename(eval_path)] = eval_data

                print("Eval Data file {0} loaded, sample num = {1}".format(eval_path, len(eval_data)))
        else:
            for eval_path in config["test_data_path"].split("|"):
                eval_data = SQLDataset(eval_path, config, hydra_featurizer, True)
                self.eval_data[os.path.basename(eval_path)] = eval_data


    def _eval_imp(self, eval_data: SQLDataset, get_sq=True):
        items = ["overall", "agg", "sel", "wn", "wc", "op", "val"]
        acc = {k:0.0 for k in items}
        agg_true = []
        agg_pred = []
        sel_true = []
        sel_pred = []
        op_true = []
        op_pred = []
        val_true = []
        val_pred = []
        wn_true = []
        wn_pred = []
        wc_true = []
        wc_pred = []
        
        sq = []
        cnt = 0
        # print("before_inf")
        model_outputs = self.model.dataset_inference(eval_data)
        # print("after_inf")
        for input_feature, model_output in zip(eval_data.input_features, model_outputs):
            cur_acc = {k:1 for k in acc if k != "overall"}

            select_label = np.argmax(input_feature.select)
            agg_label = input_feature.agg[select_label]
            wn_label = input_feature.where_num[0]
            wc_label = [i for i, w in enumerate(input_feature.where) if w == 1]

            agg, select, where, conditions = self.model.parse_output(input_feature, model_output, wc_label)
        
            if agg != agg_label:
                cur_acc["agg"] = 0
            agg_true.append(agg_label)
            agg_pred.append(agg)
            if select != select_label:
                cur_acc["sel"] = 0
            sel_true.append(select_label)
            sel_pred.append(select)
            if len(where) != wn_label:
                cur_acc["wn"] = 0
            wn_true.append(wn_label)
            wn_pred.append(len(where))
            if set(where) != set(wc_label):
                cur_acc["wc"] = 0
            wc_true.append(str(set(wc_label)))
            wc_pred.append(str(set(where)))

            for w in wc_label:
                _, op, vs, ve = conditions[w]
                if op != input_feature.op[w]:
                    cur_acc["op"] = 0
                op_true.append(input_feature.op[w])
                op_pred.append(op)

                if vs != input_feature.value_start[w] or ve != input_feature.value_end[w]:
                    cur_acc["val"] = 0
                val_true.append(str((input_feature.value_start[w], input_feature.value_end[w])))
                val_pred.append(str((vs, ve)))

            for k in cur_acc:
                acc[k] += cur_acc[k]

            all_correct = 0 if 0 in cur_acc.values() else 1
            acc["overall"] += all_correct

            if ("DEBUG" in self.config or get_sq) and not all_correct:
                try:
                    true_sq = input_feature.output_SQ()
                    pred_sq = input_feature.output_SQ(agg=agg, sel=select, conditions=[conditions[w] for w in where])
                    task_cor_text = "".join([str(cur_acc[k]) for k in items if k in cur_acc])
                    sq.append([str(cnt), input_feature.question, "|".join([task_cor_text, pred_sq, true_sq])])
                except:
                    pass
            cnt += 1

        result_str = []
        for item in items:
            result_str.append(item + ":{0:.1f}".format(acc[item] * 100.0 / cnt))

        result_str = ", ".join(result_str)
        feval = open("eval_out", "w")
        feval.write("agg: ")
        feval.write(metrics.classification_report(agg_true, agg_pred, digits=4))
        feval.write("sel: ")
        feval.write(metrics.classification_report(sel_true, sel_pred, digits=4))
        feval.write("wn: ")
        feval.write(metrics.classification_report(wn_true, wn_pred, digits=4))
        feval.write("wc: ")
        feval.write(metrics.classification_report(wc_true, wc_pred, digits=4))
        feval.write("op: ")
        feval.write(metrics.classification_report(op_true, op_pred, digits=4))
        feval.write("val: ")
        feval.write(metrics.classification_report(val_true, val_pred, digits=4))
        feval.close()
        return result_str, sq
    
    def pred_eval(self, eval_data: SQLDataset, info):
        agg_ops = ['NA', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']
        model_outputs = self.model.dataset_inference(eval_data)
        for input_feature, model_output in zip(eval_data.input_features, model_outputs):
            agg, select, where, conditions = self.model.parse_output(input_feature, model_output)
            pred_sq = input_feature.output_SQ(agg=agg, sel=select, conditions=[conditions[w] for w in where])
            # ww=0
            # while(ww<10):
            #     ww+=1
            pred_sq_list = pred_sq.split(", ")
            if info[0] != -1:
                agg = agg_ops[info[0]]
            else:
                agg = pred_sq_list[0]
            if info[1] != -1:
                cond = cond_ops[info[1]]
            else:
                cond = cond_ops[0]
            print("SQL output: ")
            if agg != "NA":
                sql_out = "SELECT "+agg+"("+pred_sq_list[1]+") from table where "+pred_sq_list[2].replace(cond_ops[0], cond)
            else:
                sql_out = "SELECT "+pred_sq_list[1]+" from table where "+pred_sq_list[2].replace(cond_ops[0], cond)
            print(sql_out)
            print("\n-------------------")
            return sql_out
            
    def eval(self, epochs, info=-1):
        # print(self.bad_case_dir)
        for eval_file in self.eval_data:
            if self.note != "":
                return self.pred_eval(self.eval_data[eval_file], info)
            result_str, sq = self._eval_imp(self.eval_data[eval_file])
            print(eval_file + ": " + result_str)

            if "DEBUG" in self.config:
                for text in sq:
                    print(text[0] + ":" + text[1] + "\t" + text[2])
            else:
                with open(self.eval_history_file, "a+", encoding="utf8") as f:
                    f.write("[{0}, epoch {1}] ".format(eval_file, epochs) + result_str + "\n")

                bad_case_file = os.path.join(self.bad_case_dir,
                                           "{0}_epoch_{1}.log".format(eval_file, epochs))
                with open(bad_case_file, "w", encoding="utf8") as f:
                    for text in sq:
                        f.write(text[0] + ":" + text[1] + "\t" + text[2] + "\n")