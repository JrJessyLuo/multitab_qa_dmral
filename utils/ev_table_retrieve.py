import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np


def read_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def get_p_r_f1(true_positives, false_positives, false_negatives, topk=None):
    if topk is None:
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives != 0
            else 0
        )
    else:
        precision = true_positives / topk

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives != 0
        else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall != 0
        else 0
    )

    return np.array([precision, recall, f1_score])


class Metrics:
    def __init__(self, top_k=None):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.acc = []
        self.perfect_recall = []
        self.p_r_f1 = np.array([0.0, 0.0, 0.0])
        self.top_k = top_k
        self.rank = []

    def update(self, preds: list[list[str]], target: list[list[str]]):
        assert len(preds) == len(target)

        cnt = 0

        for pred, tgt in zip(preds, target):
            pred = [x.upper() for x in pred]
            tgt = [x.upper() for x in tgt]

            avg_rank = np.mean([
                pred.index(t) + 1 if t in pred else float(20)
                for t in tgt
            ])
            self.rank.append(avg_rank)

            pred_set = set(pred)
            tgt_set = set(tgt)

            tp = len(pred_set & tgt_set)
            fp = len(pred_set - tgt_set)
            fn = len(tgt_set - pred_set)

            self.tp += tp
            self.fp += fp
            self.fn += fn

            if self.top_k is not None:
                self.p_r_f1 += get_p_r_f1(tp, fp, fn, self.top_k)
            else:
                self.p_r_f1 += get_p_r_f1(tp, fp, fn)

            self.acc.append(int(pred_set == tgt_set))
            self.perfect_recall.append(int(pred_set.issuperset(tgt_set)))

            cnt += 1

        if cnt == 0:
            self.p_r_f1 = np.array([0.0, 0.0, 0.0])
            self.acc = 0.0
            self.perfect_recall = 0.0
            return

        self.p_r_f1 = np.round(100 * self.p_r_f1 / cnt, 1)
        self.acc = 100 * np.array(self.acc).mean()
        self.perfect_recall = 100 * np.array(self.perfect_recall).mean()

    def precision(self):
        denominator = self.tp + self.fp
        return 0 if denominator == 0 else self.tp / denominator * 100

    def recall(self):
        denominator = self.tp + self.fn
        return 0 if denominator == 0 else self.tp / denominator * 100

    def f1(self):
        denominator = 2 * self.tp + self.fp + self.fn
        return 0 if denominator == 0 else 2 * self.tp / denominator * 100


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        obj = obj[0]
    return obj


def get_prediction_list(retrieved_tabs, topk):
    if isinstance(retrieved_tabs, dict) and "multi_hop" in retrieved_tabs:
        return list(retrieved_tabs["multi_hop"][:topk])
    return list(retrieved_tabs[:topk])

def transform(name):
    parts = name.split('#sep#')
    if len(parts) >= 3:
        return '#sep#'.join(parts[1:]) 
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--topk", required=True, type=int)
    parser.add_argument("--missing_tab_fpath", required=True, type=str)
    parser.add_argument("--qtab_fpath", required=True, type=str)
    parser.add_argument("--save_fpath", required=True, type=str)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    topk = args.topk

    q_table_map = load_pickle(args.qtab_fpath)

    missing_tab_map = {}
    if os.path.exists(args.missing_tab_fpath):
        missing_tab_map = load_pickle(args.missing_tab_fpath)
    else:
        print(f"Warning: missing_tab_fpath does not exist: {args.missing_tab_fpath}")

    question_json_fpath = f"../../dataset/data/{dataset_name}/dev.json"

    question_json = read_json(question_json_fpath)

    label_csv_path = f"../../dataset/label/{dataset_name}.csv"


    data_df = pd.read_csv(label_csv_path)

    current_infos = {}
    for qitem in question_json:
        question = qitem["question"]
        rows = data_df[data_df["question"] == question]["related_tabs"].values.tolist()

        if not rows:
            continue

        related_tabs = rows[0].split(" [SEP] ")
        current_infos[question] = {"gt": related_tabs}

    predicted_tabs = []
    all_related_tabs = []

    for question, retrieved_tabs in q_table_map.items():
        if question not in current_infos:
            continue

        singlehop_tabs = get_prediction_list(retrieved_tabs, topk)

        
        if question in missing_tab_map and missing_tab_map[question]:
            for tab in missing_tab_map[question]:
                if tab not in singlehop_tabs:
                    singlehop_tabs.append(tab)

        predicted_tabs.append([transform(_) for _ in singlehop_tabs])
        all_related_tabs.append(current_infos[question]["gt"])


    metrics = Metrics(top_k=topk)
    metrics.update(predicted_tabs, all_related_tabs)

    p, r, f1 = metrics.p_r_f1

    result = {
        "Dataset": dataset_name,
        "TopK": topk,
        "NumQuestions": len(predicted_tabs),
        "Precision": round(float(p), 2),
        "Recall": round(float(r), 2),
        "F1": round(float(f1), 2),
    }

    df = pd.DataFrame([result])
    print(df)


if __name__ == "__main__":
    main()