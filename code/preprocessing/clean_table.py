import pandas as pd
import glob
import tqdm
from tqdm import tqdm
from collections import defaultdict
import re
import json
import argparse
from uuid import uuid4
import utils
from utils import prompt, model_config, tool

recover_column_headers = recover_column_headers_prompt | llm


##########################################
#               FUNCTIONS
##########################################
def run_structured_prompt(title_name, headers, sampled_rows):
    response = recover_column_headers.invoke({"title_name": title_name, "headers":headers, "sampled_rows": sampled_rows})
    return response.content.strip()


def remove_trailing_number(s):
    # Match ending digits and remove them if found
    return re.sub(r'\d+$', '', s)

def serialize_schema(table_name, schema):
    return f"Table: {table_name}\nColumns: {schema}"



def process_dataset(dataset_name):
    fls = glob.glob(f' ../../dataset/datalake/{dataset_name}/*.csv')

    # TODO detect the tables with missing columns and headers and infer the missing ones


def main():
    parser = argparse.ArgumentParser(description="Process table files and recover metadata for a given dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'spider')")
    args = parser.parse_args()

    process_dataset(args.dataset)


if __name__ == "__main__":
    main()