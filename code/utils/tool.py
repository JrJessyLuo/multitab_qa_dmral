import duckdb
import json
import pickle
import os
import torch
import tqdm
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def read_json(fn):
  with open(fn) as f:
    return json.load(f)

def decompose_schema(tables):
  r = []
  for t in tables:
    cur_r = []
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for t_col in t_cols:
      cur_r.append(f'{t_name}:{t_col}')
    r.append(cur_r)
  return r

def decompose_schema_full(tables):
  r, col_nums = [], []
  map_dict = {}
  for t in tables:
    cur_r = []
    f_t_name, f_t_cols = tables[t]['table_name'], tables[t]['column_names']
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for f_t_col, t_col in zip(f_t_cols, t_cols):
      cur_r.append(f'{f_t_name}:{f_t_col}')
      map_dict[f'{f_t_name}:{f_t_col}'] = f'{t_name}:{t_col}'
    col_nums.append(len(t_cols))
    r.append(cur_r)
  return r, col_nums, map_dict
  
def execute_sql_on_dataframes(sql_query, tab_df_dict):
    con = duckdb.connect()
    try:
        for table_name, df in tab_df_dict.items():
            clean_name = table_name.replace('#sep#', '')
            con.register(clean_name, df)

        result_df = con.execute(sql_query).fetchdf()
        return result_df.iloc[0, 0], None  # Return result and no error

    finally:
        con.close()

# identify the tables with joinability
def identify_overlap_tables(jaccard_json_path, unique_json_path, csim_json_path, cemb_sim_json_path, dataset_name, sim_threshold=0.90, unique_threshold=0.98):
    tmp_str = json.load(open(jaccard_json_path, 'r'))
    unique_str = json.load(open(unique_json_path, 'rb'))
    csim_str = json.load(open(csim_json_path, 'rb'))
    cembsim_str = json.load(open(cemb_sim_json_path, 'rb'))
    valid_tab_pairs = []
    # cemb_threshold_dict = {'spiderdl': 0.55, 'birddl':0.65, 'test':0.55}
    cemb_threshold_dict = {'spiderwild': 0.55, 'birdwild':0.65, 'test':0.55}

    for key, tab_pair_items in tmp_str.items():
        for tab_pair, score in tab_pair_items.items():
            
            if score>sim_threshold:    
                tab_arrs = tab_pair.split('-')
                if len(tab_arrs)!=2:
                    if '-'.join(tab_arrs[:2]) in unique_str:
                        unique_ratio = max(unique_str['-'.join(tab_arrs[:2])], unique_str['-'.join(tab_arrs[2:])])
                    elif '#sep#'.join(tab_arrs[:2]) in unique_str:
                        unique_ratio = max(unique_str['#sep#'.join(tab_arrs[:2])], unique_str['#sep#'.join(tab_arrs[2:])])
                    else:
                        unique_ratio = -1
                        for i in range(3, len(tab_arrs)):
                          if '#sep#'.join(tab_arrs[:i]) in unique_str:
                            unique_ratio = max(unique_str['#sep#'.join(tab_arrs[:i])], unique_str['#sep#'.join(tab_arrs[i:])])
                        if unique_ratio==-1:
                          print('error', tab_arrs)
                          continue
                        # unique_ratio = max(unique_str[tab_arrs[0]], unique_str['-'.join(tab_arrs[1:])])
                else:
                    unique_ratio = max(unique_str[tab_arrs[0]], unique_str[tab_arrs[1]])
                if unique_ratio<unique_threshold:continue
                try:
                  left_col_key, right_col_key = tab_pair.replace(' - ', '-').split('-')
                except:
                  print('error-- when splitting column names')
                  continue
                # old code-- col_sim = csim_str[key][tab_pair]
                # col_sim whether is csim_str[left_col_key][right_col_key] or csim_str[right_col_key][left_col_key] or 0
                col_sim = 0
                if left_col_key in csim_str and right_col_key in csim_str[left_col_key]:
                    col_sim = csim_str[left_col_key][right_col_key]
                elif right_col_key in csim_str and left_col_key in csim_str[right_col_key]:
                    col_sim = csim_str[right_col_key][left_col_key] 

                  
                if dataset_name == 'test':
                  # if col_sim>0:
                    # print('bbbb')
                    # # print(cemb_sim)
                    # print(col_sim)
                    # print(score)
                    # print(tab_pair)
                    # print('---------')  
                  if col_sim<0.5:continue
                else:
                  if col_sim<0.5:continue
              
                cemb_sim = 0
                if left_col_key in cembsim_str and right_col_key in cembsim_str[left_col_key]:
                    cemb_sim = cembsim_str[left_col_key][right_col_key]
                elif right_col_key in cembsim_str and left_col_key in cembsim_str[right_col_key]:
                    cemb_sim = cembsim_str[right_col_key][left_col_key]  

                # if cemb_sim>0:
                #   print(cemb_sim)
                #   print(col_sim)
                #   print(score)
                #   print(tab_pair)
                #   print('---------')     
                  
                # cemb_sim = cembsim_str[key][tab_pair]
                # if '#sep#satscores#' in tab_pair and '#sep#frpm#' in tab_pair:
                #     print(tab_pair, score, col_sim, cemb_sim)
                
                if cemb_sim<cemb_threshold_dict[dataset_name]:continue
                

                all_tab_names = []
                for i in range(len(tab_arrs)):
                    # all_tab_names.append(tab_arrs[i])
                    if '#sep#' in tab_arrs[i]:
                        all_tab_names.append('#sep#'.join( tab_arrs[i].split('#sep#')[:-1] ))
                if len(all_tab_names) != 2:
                    print('error ---')
                    print(all_tab_names, tab_arrs, tab_pair)
                    continue            
                # assert len(all_tab_names) == 2
                
                updated_tab_pair = sorted(all_tab_names)
                if updated_tab_pair not in valid_tab_pairs:
                    valid_tab_pairs.append(updated_tab_pair)

    return valid_tab_pairs


def extract_json_output(llm_response):
  try:
      json_output = json.loads(llm_response)
  except:
      try:
          json_output =  ast.literal_eval(llm_response)
      except:
          print(f'json read error')
          json_output = None
  return json_output
          

BATCH_SIZE = 200
def embed(texts, fn, hide_progress=False):
  if fn is not None and os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))
  
  tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
  model = AutoModel.from_pretrained('facebook/contriever-msmarco').cuda()

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1), disable=hide_progress):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if len(_texts) == 0:
      break
    assert len(_texts) >= 1
    
    inputs = tokenizer(_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
      vec = model(**inputs)
    vec = mean_pooling(vec[0], inputs['attention_mask'])

    embeds.append(vec.cpu().numpy())
  
  embeds = np.vstack(embeds)

  if fn is not None:
    np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds