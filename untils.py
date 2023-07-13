import pandas as pd
import glob
import os
import time
import datetime
from typing import List, Union, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from text2vec.utils.rank_bm25 import BM25Okapi
from text2vec.utils.tokenizer import JiebaTokenizer

new_file_ID = [104, 136, 297, 490, 639, 914, 1375,
               1502, 1732, 1757, 1864, 2795, 2906,
               3732, 3989]

# 所有数据列
all_columns = ['时间衰减系数', '点赞-mean', '点赞-max', '点赞-var', '点赞-sum',
                '转发-mean', '转发-max', '转发-var', '转发-sum',
                '评论-mean', '评论-max', '评论-var', '评论-sum',
                '粉丝数-mean', '粉丝数-max', '粉丝数-var', '粉丝数-sum',
                '关注数-mean', '关注数-max', '关注数-var', '关注数-sum',
                '全文内容-词频相似性-mean', '全文内容-词频相似性-max', '全文内容-词频相似性-var', '全文内容-词频相似性-sum',
                '标题/微博内容-词频相似性-mean', '标题/微博内容-词频相似性-max', '标题/微博内容-词频相似性-var', '标题/微博内容-词频相似性-sum',
                '全文内容-语义相似性-mean', '全文内容-语义相似性-max', '全文内容-语义相似性-var', '全文内容-语义相似性-sum',
                '标题/微博内容-语义相似性-mean', '标题/微博内容-语义相似性-max', '标题/微博内容-语义相似性-var', '标题/微博内容-语义相似性-sum',
                '全文内容-情感极性值-mean', '全文内容-情感极性值-max', '全文内容-情感极性值-var', '全文内容-情感极性值-sum',
                '标题/微博内容-情感极性值-mean', '标题/微博内容-情感极性值-max', '标题/微博内容-情感极性值-var', '标题/微博内容-情感极性值-sum']

# 标量型数据列
int64_columns = ['点赞-mean', '点赞-max', '点赞-var', '点赞-sum',
                '转发-mean', '转发-max', '转发-var', '转发-sum',
                '评论-mean', '评论-max', '评论-var', '评论-sum',
                '粉丝数-mean', '粉丝数-max', '粉丝数-var', '粉丝数-sum',
                '关注数-mean', '关注数-max', '关注数-var', '关注数-sum']

# 词频（词面量）相似性、语义相似性 数据列
semantic_columns = ['全文内容-词频相似性-mean', '全文内容-词频相似性-max', '全文内容-词频相似性-var', '全文内容-词频相似性-sum',
                    '标题/微博内容-词频相似性-mean', '标题/微博内容-词频相似性-max', '标题/微博内容-词频相似性-var', '标题/微博内容-词频相似性-sum',
                    '全文内容-语义相似性-mean', '全文内容-语义相似性-max', '全文内容-语义相似性-var', '全文内容-语义相似性-sum',
                    '标题/微博内容-语义相似性-mean', '标题/微博内容-语义相似性-max', '标题/微博内容-语义相似性-var', '标题/微博内容-语义相似性-sum']

# 情感数据列
emotion_columns = ['全文内容-情感极性值-mean', '全文内容-情感极性值-max', '全文内容-情感极性值-var', '全文内容-情感极性值-sum',
                    '标题/微博内容-情感极性值-mean', '标题/微博内容-情感极性值-max', '标题/微博内容-情感极性值-var', '标题/微博内容-情感极性值-sum']


# 将 './data/events' 下所有数据的 某一列-'全文内容' 合并为一个语料库
def combine_content(file_path = './data/events/*.xlsx', column_name = '全文内容', save_path = './data/word2vec/contents_corpus.csv'):
    # 获取所有符合条件的文件路径
    files = glob.glob(file_path)
    # 读取每个文件中指定列的数据，并将数据合并为一个 DataFrame
    dataframes = []
    for file in files:
        try:
            df = pd.read_excel(file, usecols=[column_name])
            dataframes.append(df)
        except:
            continue
    merged_df = pd.concat(dataframes, ignore_index=True)
    # 将 DataFrame 中的数据转换为一个列表
    print('语料库长度：', len(merged_df))
    merged_df.to_csv(save_path, header=True)


# 读取总文件
def read_events_heat(dir_path='./data', filename="heat_events.xlsx"):
    events_heat_path = os.path.join(dir_path, filename)  # 所有时间热度的excel
    all_events_heat = pd.read_excel(events_heat_path)
    return all_events_heat

# 读取单个事件的文件
def read_sample_event(ID, dir_allData_path='./data/events'):
    current_sample_path = os.path.join(dir_allData_path, str(ID) + '.xlsx')
    sample_event = pd.read_excel(current_sample_path)
    return sample_event

# 计算column列的平均值、最大值、方差
def calcute_mean_max_var(df, column):
    mean_value = df[column].mean()
    max_value = df[column].max()
    var_value = df[column].var()
    sum_value = df[column].sum() 
    return mean_value, max_value, var_value, sum_value

# 将 点赞|转发|评论|粉丝数|关注数 这些标量值 的额外数值特征算出来
def calculate_int64_extra(df, cal_type=None):
    if cal_type is None:
        cal_type = ['点赞', '转发', '评论', '粉丝数', '关注数']
    re_arr = []
    for column in cal_type:
        mean_value, max_value, var_value,sum_value = calcute_mean_max_var(df, column)
        re_arr.extend([mean_value, max_value, var_value,sum_value])
    return re_arr





# district_dict
def make_district_dict():
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    # 注意还有 海外地区
    district_dict=le.fit([
        "河北","山西",
        "黑龙江","吉林",
        "辽宁","江苏",
        "浙江","安徽",
        "福建","江西",
        "山东","河南",
        "湖北","湖南",
        "广东","海南",
        "四川","贵州",
        "云南","陕西",
        "甘肃","青海","台湾",
        "内蒙古","广西","西藏","宁夏","新疆",
        "北京","天津","上海","重庆",
        "香港","澳门","海外","其他"
    ])
    print("district_dict classes:",district_dict.classes_)
    
    # district_dict demo
    # print(district_dict.transform(["广东","四川"]))

    return district_dict


# 136 297 490 496 639 914 996 1375 1502 1732 1757 1864 2795 2906 3451 3732 3989
def calculate_int64_extra_district(dir_allData_path='./data/events'):
    all_events_heat = read_events_heat()
    district_dict = make_district_dict()
    IDs = all_events_heat['序号'].values
    # 保存的list
    save_arr = []
    
    for ID in IDs:
        sample_event_path = os.path.join(dir_allData_path, str(ID)+'.xlsx')
        try:
            sample_event = pd.read_excel(sample_event_path)
            # 数值标量统计信息
            extra_arr = sample_event["地域"].value_counts()
            types_num = extra_arr.values.shape[0]
            index_list = district_dict.classes_
            templete_arr = []
            for district in index_list:
                if district in extra_arr.index.values:
                    templete_arr.append(extra_arr[district])
                else:
                    templete_arr.append(0)
            save_arr.append([ID, *templete_arr])
        except:
            print(ID)
            save_arr.append([ID])
    # 转excel数据
    excel_data_district= pd.DataFrame(save_arr,columns=[ID,*index_list])        
    excel_data_district.to_excel(f"./data/{time.time()}_district_feature_data.xlsx")
    return save_arr,index_list

# 余弦相似度计算
def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# 自定义语义搜索函数，返回不排序的语义相似度
def semantic_search(query_embeddings: Tensor,
                    corpus_embeddings: Tensor,
                    query_chunk_size: int = 100,
                    corpus_chunk_size: int = 500000,
                    top_k: int = 10,
                    score_function=cos_sim,
                    if_soft=False):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    query_embeddings = query_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarity
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    # Sort and strip to top_k results
    if if_soft:
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list

    return queries_result_list



def calculate_days_diff(save_status=False,save_path="./data"):
    nowadays = "2023-06-25 00:00:00"
    # timeArray = time.strptime(nowadays, "%Y-%m-%d %H:%M:%S")
    datetimeArray=datetime.datetime.strptime(nowadays, "%Y-%m-%d %H:%M:%S")
    all_events_heat = read_events_heat()
    all_events_heat["时间衰减系数"] = (datetimeArray-all_events_heat["起始时间"]).apply(lambda x: x.days/365)
    if save_status == True:
        filename = "extra_4.xlsx"
        all_events_heat.to_excel(save_path+"/"+filename, index=False)
        
    return all_events_heat


# data preprocess section 2
# 解决评论的地域编码
# def 