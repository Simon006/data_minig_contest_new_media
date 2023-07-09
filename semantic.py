import sys
import os

import numpy
from text2vec import SentenceModel
from similarities.literalsim import TfidfSimilarity, BM25Similarity
import torch

from untils import read_events_heat, read_sample_event, cos_sim, semantic_search

embedder = SentenceModel("shibing624/text2vec-base-chinese-paraphrase")


# 测试API
def test():
    # Corpus with example sentences
    corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
    ]
    corpus_embeddings = embedder.encode(corpus)

    # Query sentences:
    queries = [
        '如何更换花呗绑定银行卡',
        '敏捷的棕色狐狸跳过了懒狗',
    ]

    # print('#' * 42)
    # ########  use semantic_search to perform cosine similarty + topk
    # print('\nuse semantic_search to perform cosine similarty + topk:')
    #
    # for query in queries:
    #     query_embedding = embedder.encode(query)
    #     hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
    #     print("\n\n======================\n\n")
    #     print("Query:", query)
    #     print("\nTop 5 most similar sentences in corpus:")
    #     hits = hits[0]  # Get the hits for the first query
    #     for hit in hits:
    #         print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

    print('#' * 42)
    ######## use bm25 to rank search score
    print('\nuse bm25 to calc each score:')

    m = BM25Similarity()
    m.add_corpus(corpus)
    res = m.most_similar(queries)
    print('sim search: ', res)
    for q_id, c in res.items():
        print('query:', queries[q_id])
        for corpus_id, s in c.items():
            print(f'\t{m.corpus[corpus_id]}: {s:.4f}')



# 计算文本 内容集contents 与 标题title 的 Tfidf
# return：list scores
def cal_TfidfSimilarity(contents, title, if_corpus = False):
    print('\nuse Tfidf to calc each score:')
    search_sim = TfidfSimilarity()

    if if_corpus:                                       # 是否根据整个语料库构建词频向量 / 单独两个句子的对应词频
        corpus = contents + [title]
        try:
            search_sim.add_corpus(corpus)
        except:
            pass

    try:
        Tfidf_scores = search_sim.similarity(title, contents)
        Tfidf_scores_list = Tfidf_scores.numpy().tolist()
        # 只有一个query
        return Tfidf_scores_list[0]
    except:
        return [0] * len(contents)


# 计算 内容集contents 与 标题title 的 语义相似度
# return: hits    hit['corpus_id'] 对应内容文本的下标，可以直接 contents[hit['corpus_id']] 取得内容文本
#                 hit['score'] 计算得到的对应的语义相似度
def cal_semantic_similarity(contents, title):
    print('\nuse semantic_search to perform cosine similarity')
    try:
        contents_embeddings = embedder.encode(contents)
        test_title_embedding = embedder.encode(title)
        hits = semantic_search(test_title_embedding, contents_embeddings, top_k=10000, if_soft=False)
        hits = hits[0]          # 只有一个query
        hits_score = [hit['score'] for hit in hits]
        return hits_score
    except:
        return [0] * len(contents)

# 根据 'heat_events' 中 '标题/微博内容' 、 '全文内容' 和 '事件' 计算语义相似度\Tfidf
def cal_semantic():
    if not os.path.exists('./data/events_semantic'):
        os.makedirs('./data/events_semantic')

    all_events = read_events_heat()
    titles = list(all_events['事件'].values)
    IDs = list(all_events['序号'].values)

    # ID对应events文件的文件名
    for i in range(len(IDs)):
        ID = IDs[i]
        title = titles[i]

        # 某些文件读取失败
        try:
            sample = read_sample_event(ID)
        except:
            continue

        contents = list(sample['全文内容'].values)
        contents_short = list(sample['标题/微博内容'].values)

        # 根据每件事件与转发文本，计算词频相似性   (转发文本与原事件文本的相似性)
        contents_Tfidf_score = cal_TfidfSimilarity(contents, title, if_corpus=True)
        contents_short_Tfidf_score = cal_TfidfSimilarity(contents_short, title, if_corpus=True)

        # 根据每件事件与转发文本，计算语义相似性   (转发文本与原事件文本的相似性)
        contents_hits = cal_semantic_similarity(contents, title)
        contents_short_hits = cal_semantic_similarity(contents_short, title)

        print(len(contents_Tfidf_score), len(contents_hits), len(contents_short_Tfidf_score), len(contents_short_hits))

        sample['全文内容-词频相似性'] = contents_Tfidf_score
        sample['标题/微博内容-词频相似性'] = contents_short_Tfidf_score
        sample['全文内容-语义相似性'] = contents_hits
        sample['标题/微博内容-语义相似性'] = contents_short_hits

        sample.to_excel('./data/events_semantic/' + str(ID) + '.xlsx', index=False)




if __name__ == '__main__':
    # test()
    cal_semantic()