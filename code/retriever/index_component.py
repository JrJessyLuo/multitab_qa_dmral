import utils
from utils.model_config import embedder
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from typing import List, Dict
import json
import numpy as np
import pickle
import tqdm
import json
import string
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))
stop_words.update(['name', 'id'])
# import spacy

# -- Build FAISS Vector Store from Multiple Papers --
# def is_leaf_section(section):
#     return not section.get("section_children")

# def build_faiss_index(paper_section_map: Dict[str, List[Dict]], save_path="./faiss_store"):
#     texts = []
#     metadatas = []
#     for paper_id, sections in paper_section_map.items():
#         for s in sections:
            
#             # TODO: may be not a good idea
#             if not is_leaf_section(s):
#                 continue
            
#             text = s["section_summary"] or s["section_content"][:500]
#             metadata = {
#                 "paper_id": paper_id,
#                 "section_id": s["section_id"],
#                 "title": s["section_title"],
#                 "tags": s.get("section_tag", []),
#                 "section_level": s.get("section_level"),
#                 "section_parent": s.get("section_parent"),
#                 "section_children": s.get("section_children", [])
#             }
#             texts.append(text)
#             metadatas.append(metadata)

#     faiss_index = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
#     faiss_index.save_local(save_path)
#     return faiss_index


def build_faiss_index(table_schema_map, save_path="./faiss_store"):
    texts = []
    metadatas = []
    for table_name, table_schema in table_schema_map.items():
        texts.append(table_schema)
        metadatas.append({"table_name": table_name})

    faiss_index = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
    if save_path is not None:
      faiss_index.save_local(save_path)
    return faiss_index




# -- Optional: Predict Tags by Embedding Similarity (with definitions) --
# def predict_tags(question: str, top_k=3):
#     tag_texts = []
#     tag_names = []

#     for k, v in TAG_DEFINITIONS.items():
#         tag_texts.append(f"{k}: {v}")
#         tag_names.append(k)
    
#     tag_embeddings = embedder.embed_documents(tag_texts)

#     question_emb = embedder.embed_query(question)
#     sims = cosine_similarity([question_emb], tag_embeddings)[0]
#     return [tag_names[i] for i in np.argsort(sims)[-top_k:][::-1]]


# # -- Hybrid Section Retriever --
# def retrieve_tables(question, faiss_index, top_k=20):
#     results = faiss_index.similarity_search_with_score(question, k=top_k)

#     def compute_score(res):
#         score = res[1]
#         return score

#     reranked = sorted(results, key=compute_score, reverse=True)[:top_k]
#     return [(_[0].page_content, _[1])  for _ in reranked]

# def retrieve_tables(question, faiss_index, top_k=20):
#     # Step 1: FAISS fast retrieval (approximate)
#     results = faiss_index.similarity_search_with_score(question, k=top_k)

#     # Step 2: Embed the query
#     query_vec = np.array(embedder.embed_query(question)).reshape(1, -1)

#     # Step 3: Compute cosine similarity for re-ranking
#     reranked = []
#     for doc, _ in results:
#         # doc_embedding = np.array(embedder.embed_query(doc.page_content)).reshape(1, -1)  # same embedder used for docs
#         doc_embeds = embedder.embed_documents([doc.page_content])  # make sure this returns a list of vectors
#         doc_embedding = np.mean(np.array(doc_embeds), axis=0).reshape(1, -1)

#         cos_sim = float(cosine_similarity(query_vec, doc_embedding)[0][0])

#         reranked.append((doc.page_content, cos_sim))

#     # Step 4: Sort by cosine similarity
#     reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

#     return reranked
# update it
def retrieve_tables(question, faiss_index, top_k=20):
    # Step 1: FAISS fast retrieval (approximate)
    results = faiss_index.similarity_search_with_score(question, k=top_k)

    # Step 2: Embed the query
    query_vec = np.array(embedder.embed_query(question)).reshape(1, -1)

    # Step 3: Compute cosine similarity for re-ranking
    reranked = []
    for doc, _ in results:
        # doc_embedding = np.array(embedder.embed_query(doc.page_content)).reshape(1, -1)  # same embedder used for docs
        doc_embeds = embedder.embed_documents([doc.page_content])  # make sure this returns a list of vectors
        doc_embedding = np.mean(np.array(doc_embeds), axis=0).reshape(1, -1)

        cos_sim = float(cosine_similarity(query_vec, doc_embedding)[0][0])

        reranked.append((doc.page_content, cos_sim, doc.metadata))

    # Step 4: Sort by cosine similarity
    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

    return reranked

def retrieve_tables_bird(question, faiss_index, all_documents, embedder, top_k=20, keywords=None):
    """
    Step 1: Filter candidate documents using keyword search.
    Step 2: Run similarity search and re-rank with cosine similarity.

    Parameters:
        question (str): input question
        faiss_index: FAISS index
        all_documents (List[Document]): original list of documents in the index
        embedder: embedding model with `embed_query` and `embed_documents`
        top_k (int): number of top results to return
        keywords (List[str]): list of keywords for filtering

    Returns:
        List of (doc_content, cosine_score) tuples
    """

    # Step 1: Keyword filtering (before FAISS)
    if keywords:
        filtered_docs = {
            key:doc for key, doc in all_documents.items()
            if any(kw in doc.split() for kw in keywords)
        }
    else:
        filtered_docs = all_documents

    if not filtered_docs:
        return []  # No candidates to search


    filtered_faiss = create_faiss_index(filtered_docs, None)

    results = filtered_faiss.similarity_search_with_score(question, k=min(top_k, len(filtered_docs)))

    # Step 3: Re-rank using cosine similarity
    query_vec = np.array(embedder.embed_query(question)).reshape(1, -1)

    reranked = []
    for doc, _ in results:
        doc_embeds = embedder.embed_documents([doc.page_content])
        doc_embedding = np.mean(np.array(doc_embeds), axis=0).reshape(1, -1)
        cos_sim = float(cosine_similarity(query_vec, doc_embedding)[0][0])
        # reranked.append((doc.page_content, cos_sim))
        reranked.append((doc.page_content, cos_sim, doc.metadata))

    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# def load_sections(json_path: str) -> List[Dict]:
#     with open(json_path) as f:
#         return json.load(f)["entity_section"]
    
# def load_json_exmaples():
#     paper_map = {
#         "UAE": load_sections("sections_output.json")
#     }
#     return paper_map

def load_faiss_index(path="./faiss_store"):
    faiss_index = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    return faiss_index

def create_faiss_index(table_schema_map, save_path="./faiss_store"):
    faiss_index = build_faiss_index(table_schema_map, save_path)
    return faiss_index

def decompose_schema(tables):
  r, table_col_dictionary = [], []
  for t in tqdm.tqdm(tables):
    t_name, t_cols, t_full_cols = tables[t]['table_name_original'], tables[t]['column_names_original'], tables[t]['column_names']
    for t_col, t_full_col in zip(t_cols, t_full_cols):
      r.append(f'{t_name}:{t_full_col}')
      table_col_dictionary.append(f'{t_name}:{t_col}')
  return r, table_col_dictionary

# def test_create_entry():
#     paper_map = load_json_exmaples()
#     faiss_index = create_faiss_index(paper_map, save_path="./faiss_store")
#     print(f"FAISS index created and saved")
#     return faiss_index

def read_json(fn):
  with open(fn) as f:
    return json.load(f)

def serialize_table(table):
  db_id, table_name, cols = table['db_id'], table['table_name_original'], table['column_names_original']
  return ' '.join([db_id, table_name] + cols)

def load_all_cols(tables):
  table_col_dictionary = []
  for t in tqdm.tqdm(tables):
    t_name, t_cols, t_full_cols = tables[t]['table_name_original'], tables[t]['column_names_original'], tables[t]['column_names']
    for t_col, t_full_col in zip(t_cols, t_full_cols):
      if t_col not in table_col_dictionary:
        table_col_dictionary.append(f'{t_col}')
  return table_col_dictionary


def load_evidence(evidence_path, all_cols):
    question_evidence_map_dict = {}
    for line in json.load(open(evidence_path, 'r')):
        question = line['question']
        evidence = line['evidence']
        if len(evidence.strip())==0:continue
        
        content_col_map_dict = {}
        for content in evidence.split(';'):
          valid_cols = [_ for _ in all_cols if _.lower() in content.lower()]
          if len(valid_cols) == 0:continue
          content_col_map_dict[content] = valid_cols

        question_evidence_map_dict[question] = content_col_map_dict
          
    return question_evidence_map_dict

def filter_stopwords_nltk(word_list):
    return [word for word in word_list if word.lower() not in stop_words ]

def map_evidence_question(evidence_dict, subquestions):
    subquest_keywords = {}
    for ev_, keywords in evidence_dict.items():
      question_emb = model.encode([ev_])
      sent_embs = model.encode(subquestions)

      similarities = cosine_similarity(question_emb, sent_embs)[0] 
      max_index = np.argmax(similarities)

      most_similar_ques = subquestions[max_index]
      if most_similar_ques not in subquest_keywords:
        subquest_keywords[most_similar_ques] = []
      subquest_keywords[most_similar_ques].extend(keywords)
    return subquest_keywords

def filter_name_variants(column_list):
    lowercased = [col.lower() for col in column_list]

    name_variants = ['name']

    has_variant = any(variant in col for col in lowercased for variant in name_variants)

    if has_variant:
        # Remove exact 'name' or any trivial lowercase 'name'
        filtered = [col for col in column_list if col.lower() != 'name']
    else:
        filtered = column_list

    return filtered
