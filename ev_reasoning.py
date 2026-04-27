import os
import pickle
import pandas as pd
from tqdm import tqdm
from utils import tool
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--topk", required=True, type=int)
    parser.add_argument("--sql_path", required=True, type=str)
    parser.add_argument("--table_path", required=True, type=str)
    parser.add_argument("--missing_path", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset
    topk = args.topk

    datalake_path =  f"../dataset/data/{dataset}"

    # ---- load ----
    sql_dict = pickle.load(open(args.sql_path, "rb"))
    q_table_map = pickle.load(open(args.table_path, "rb"))
    missing_tab_map = pickle.load(open(args.missing_path, "rb"))

    if isinstance(sql_dict, list):
        sql_dict = sql_dict[0]
    if isinstance(q_table_map, list):
        q_table_map = q_table_map[0]
    if isinstance(missing_tab_map, list):
        missing_tab_map = missing_tab_map[0]

    # ---- load GT ----
    gt_path = f"../dataset/label/{dataset}.csv"

    df = pd.read_csv(gt_path)

    answer_map = {
        row["question"]: round(float(row["answer"]), 2)
        for _, row in df.iterrows()
    }

    # ---- execution ----
    cache_tables = {}
    correct = 0
    total = 0

    for q, retrieved_tabs in tqdm(q_table_map.items()):

        if q not in sql_dict:
            continue

        # ---- tables ----
        tabs = retrieved_tabs["multi_hop"][:topk]

        if q in missing_tab_map and missing_tab_map[q]:
            for t in missing_tab_map[q]:
                if t not in tabs:
                    tabs.append(t)

        # ---- load tables ----
        tab_df_dict = {}
        for t in tabs:
            key = t.replace("#sep#", "_")

            if key not in cache_tables:
                table_path = os.path.join(datalake_path, f"{t}.csv")
                cache_tables[key] = pd.read_csv(table_path)

            tab_df_dict[key] = cache_tables[key]

        # ---- sql ----
        sql = sql_dict[q]
        if isinstance(sql, dict):
            sql = sql["sql"]
        if isinstance(sql, list):
            sql = sql[0] if sql else ""

        if not sql:
            continue

        sql = sql.replace("#sep#", "_")

        # ---- execute ----
        try:
            answer, _ = tool.execute_sql_on_dataframes(sql, tab_df_dict)
            pred = round(float(answer), 2)
            gt = answer_map[q]

            if pred == gt:
                correct += 1

        except:
            pass

        total += 1

    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")


if __name__ == "__main__":
    main()