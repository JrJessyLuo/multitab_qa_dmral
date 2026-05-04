import os
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from utils import tool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--topk", required=True, type=int)
    parser.add_argument("--sql_path", required=True, type=str)
    parser.add_argument("--table_path", required=True, type=str)
    parser.add_argument("--missing_path", required=True, type=str)

    parser.add_argument(
        "--unionable",
        action="store_true",
        help="Expand retrieved tables with unionable tables."
    )
    parser.add_argument(
        "--unionable_path",
        default=None,
        type=str,
        help="Path to dev_unionable_clusters.json. "
             "Default: ../../dataset/data/{dataset}/dev_unionable_clusters.json"
    )

    return parser.parse_args()


def load_pickle(path):
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, list):
        obj = obj[0]
    return obj


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unique_preserve_order(items):
    seen = set()
    output = []

    for x in items:
        if x not in seen:
            seen.add(x)
            output.append(x)

    return output


def get_retrieved_tabs(retrieved_tabs, topk):
    """
    Support both:
    - {"multi_hop": [...]}
    - [...]
    """
    if isinstance(retrieved_tabs, dict) and "multi_hop" in retrieved_tabs:
        return list(retrieved_tabs["multi_hop"][:topk])

    return list(retrieved_tabs[:topk])


def expand_with_unionable_tables(base_tabs, unionable_items):
    """
    Expand selected tables with their unionable tables.

    final_tabs = base_tabs + unionable tables of each base table
    """
    if not unionable_items:
        return unique_preserve_order(base_tabs)

    table_to_unionable = unionable_items.get("table_to_unionable_tables", {})

    expanded_tabs = []
    for tab in base_tabs:
        expanded_tabs.append(tab)

        if tab in table_to_unionable:
            expanded_tabs.extend(table_to_unionable[tab])

    return unique_preserve_order(expanded_tabs)


def normalize_table_name(tab):
    return tab.replace("#sep#", "_")


def normalize_sql_obj(sql_obj):
    """
    Normalize saved SQL object into a SQL string.
    """
    sql = sql_obj

    if isinstance(sql, dict):
        if "sql" in sql:
            sql = sql["sql"]
        elif "SQL" in sql:
            sql = sql["SQL"]
        elif "Final SQL" in sql:
            sql = sql["Final SQL"]

    if isinstance(sql, list):
        sql = sql[0] if sql else ""

    if not isinstance(sql, str):
        sql = str(sql)

    return sql.strip()


def main():
    args = parse_args()

    dataset = args.dataset
    topk = args.topk

    # NOTE: original code used dataset_name here, but dataset_name is undefined.
    datalake_path = f"../../dataset/datalake/{dataset}/"

    # ---- load SQL / retrieved tables / missing tables ----
    sql_dict = load_pickle(args.sql_path)
    q_table_map = load_pickle(args.table_path)
    missing_tab_map = load_pickle(args.missing_path)

    # ---- load unionable tables ----
    unionable_items = None
    if args.unionable:
        unionable_path = args.unionable_path
        if unionable_path is None:
            unionable_path = f"../../dataset/data/{dataset}/dev_unionable_clusters.json"

        if not os.path.exists(unionable_path):
            raise FileNotFoundError(f"Unionable file not found: {unionable_path}")

        unionable_items = read_json(unionable_path)
        print(f"Loaded unionable tables from {unionable_path}")

    # ---- load GT answers ----
    gt_path = f"../../dataset/label/{dataset}.csv"
    df = pd.read_csv(gt_path)

    answer_map = {
        row["question"]: round(float(row["answer"]), 2)
        for _, row in df.iterrows()
    }

    # ---- execution ----
    cache_tables = {}
    correct = 0
    total = 0
    skipped = 0

    for q, retrieved_tabs in tqdm(q_table_map.items()):

        if q not in sql_dict:
            continue

        if q not in answer_map:
            continue

        # 1. top-k retrieved tables
        tabs = get_retrieved_tabs(retrieved_tabs, topk)

        # 2. add recalled missing tables
        if q in missing_tab_map and missing_tab_map[q]:
            for t in missing_tab_map[q]:
                if t not in tabs:
                    tabs.append(t)

        # 3. add unionable tables
        if args.unionable:
            tabs = expand_with_unionable_tables(tabs, unionable_items)

        # 4. de-duplicate while preserving order
        tabs = unique_preserve_order(tabs)

        # ---- load tables ----
        tab_df_dict = {}

        for t in tabs:
            key = normalize_table_name(t)

            if key not in cache_tables:
                cur_table_path = os.path.join(datalake_path, f"{t}.csv")

                if not os.path.exists(cur_table_path):
                    print(f"⚠️ Missing table file: {cur_table_path}")
                    continue

                cache_tables[key] = pd.read_csv(cur_table_path)

            tab_df_dict[key] = cache_tables[key]

        # ---- SQL ----
        sql = normalize_sql_obj(sql_dict[q])

        if not sql:
            skipped += 1
            continue

        sql = sql.replace("#sep#", "_")

        # ---- execute ----
        try:
            answer, error = tool.execute_sql_on_dataframes(sql, tab_df_dict)

            if answer == -1 or pd.isna(answer):
                skipped += 1
                print(f"❌ Execution failed for question: {q}")
                print(f"SQL: {sql}")
                print(f"Error: {error}")
                continue

            pred = round(float(answer), 2)
            gt = answer_map[q]

            if pred == gt:
                correct += 1

        except Exception as e:
            skipped += 1
            print(f"❌ Exception for question: {q}")
            print(f"SQL: {sql}")
            print(f"Error: {e}")
            continue

        total += 1

    if total == 0:
        print("Accuracy: 0/0 = 0.0000")
    else:
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")

    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()