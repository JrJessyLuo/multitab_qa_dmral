import pandas as pd
import glob
import tqdm
from tqdm import tqdm
from collections import defaultdict
import re
import os
import json
import argparse
from uuid import uuid4
# from llm_tool import run_structured_prompt
# from utils.tool import read_json, write_json

def read_csv_auto_index(path, **kwargs):
    """
    Read CSV files robustly.

    If the first column is an accidentally saved pandas index column
    such as 'Unnamed: 0', read it as index_col=0.
    Otherwise, read the CSV normally.
    """
    header = pd.read_csv(path, nrows=0, **kwargs)
    first_col = str(header.columns[0])

    if first_col.startswith("Unnamed"):
        return pd.read_csv(path, index_col=0, **kwargs)

    return pd.read_csv(path, **kwargs)

def remove_trailing_number(s):
    # Match ending digits and remove them if found
    return re.sub(r'\d+$', '', s)

def serialize_schema(table_name, schema):
    return f"Table: {table_name}\nColumns: {schema}"



def process_dataset(dataset_name):
    # Locate files
    fls = glob.glob(f'../../dataset/datalake/{dataset_name}/*.csv')

    all_dict = {}
    processed_table_names = []

    # Process each CSV file
    for f in tqdm(fls, total=len(fls)):
        updated_title_name = f.split('/')[-1][:-4]
        try:
            df = read_csv_auto_index(f)
        except:
            print(f'error {updated_title_name}')
            continue
        
        headers = list(df.columns)

        db_id = updated_title_name.split('#sep#')[0]

        column_types = [
            "number" if col in df.select_dtypes(include='number').columns else "text"
            for col in df.columns
        ]
        title_name = updated_title_name.split('#sep#')[-1]

        table_dict = {
            "db_id": db_id,
            "table_name_original": title_name,
            "table_name": ' '.join(title_name.split('_')),
            "column_names_original": headers,
            "column_names": [' '.join(_.split('_')) for _ in headers],
            "column_types": column_types
        }

        all_dict[updated_title_name] = table_dict
        # print(table_dict)

    # Save output
    output_path = f"../../dataset/data/{dataset_name}/dev_tables.json"
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as outfile:
        json.dump(all_dict, outfile, indent=2)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process table files and recover metadata for a given dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'spider')")
    args = parser.parse_args()

    process_dataset(args.dataset)


if __name__ == "__main__":
    main()