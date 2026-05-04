import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


def union_fun(dataset_name):
    mode = "dev"
    tables_json_path = f"../../dataset/data/{dataset_name}/{mode}_tables.json"

    with open(tables_json_path, "r") as f:
        tables = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    schema_embeddings = []
    table_ids = []
    table_titles = []
    column_counts = []

    def embed_schema(column_names):
        embeddings = model.encode(column_names, convert_to_tensor=True)
        return torch.mean(embeddings, dim=0)

    for tid, table_info in tqdm(tables.items(), desc="Embedding Schemas"):
        col_names = [
            name for name in table_info["column_names_original"]
            if isinstance(name, str) and name.strip()
        ]

        if not col_names:
            continue

        schema_emb = embed_schema(col_names)
        schema_embeddings.append(schema_emb.cpu().numpy())
        table_ids.append(tid)
        table_titles.append(table_info["db_id"].lower()+" "+table_info["table_name"].lower())
        column_counts.append(len(col_names))

    if not table_ids:
        raise ValueError("No valid tables found for schema embedding.")

    schema_embeddings_np = np.stack(schema_embeddings)
    similarity_matrix = cosine_similarity(schema_embeddings_np)

    threshold = 0.8
    title_threshold = 0.9

    unionable_pairs = []

    def title_similarity(title1, title2):
        return SequenceMatcher(None, title1, title2).ratio()

    for i in range(len(table_ids)):
        for j in range(i + 1, len(table_ids)):
            if (
                similarity_matrix[i, j] >= threshold
                and column_counts[i] == column_counts[j]
                and title_similarity(table_titles[i], table_titles[j]) >= title_threshold
            ):
                unionable_pairs.append((table_ids[i], table_ids[j]))

    parent = {tid: tid for tid in table_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for t1, t2 in unionable_pairs:
        union(t1, t2)

    cluster_dict = {}
    group_id = 0

    for tid in table_ids:
        root = find(tid)

        if root not in cluster_dict:
            cluster_dict[root] = {
                "group_id": f"group_{group_id}",
                "tables": []
            }
            group_id += 1

        cluster_dict[root]["tables"].append(tid)

    groups = {
        v["group_id"]: v["tables"]
        for v in cluster_dict.values()
        if len(v["tables"]) > 1
    }

    table_to_unionable_tables = {}

    for tables_in_group in groups.values():
        for tid in tables_in_group:
            table_to_unionable_tables[tid] = [
                other_tid for other_tid in tables_in_group
                if other_tid != tid
            ]

    output = {
        "groups": groups,
        "table_to_unionable_tables": table_to_unionable_tables
    }

    output_path = f"../../dataset/data/{dataset_name}/{mode}_unionable_clusters.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    union_fun(args.dataset)