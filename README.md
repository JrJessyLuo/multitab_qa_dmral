# DMRAL

This repository contains the code and benchmarks for the **DMRAL** framework, designed for numerical multi-table question answering (MTQA) in tabular data lakes.

---

## 📦 Benchmarks

We release two benchmarks to support evaluation:

- [**SpiderDL**](https://zenodo.org/records/15486949)
- [**BirdDL**](https://zenodo.org/records/15488031)


Each benchmark simulates realistic multi-table analytical reasoning tasks in data lakes by transforming existing text-to-SQL datasets.

---

## 🚀 Running the DMRAL Framework

### 1. Installation

Create and activate a conda environment, then install all required Python dependencies:

```bash
conda create -n dmral_env python=3.8 -y
conda activate dmral_env
pip install -r requirements.txt
```

### 2. Running the Full Pipeline
> **Note**: To run the code on the **BirdDL** dataset, make sure to update the `dataset_name` variable in `bash.sh` to `birddl`.

```bash
cd code
bash bash.sh
```
