import jieba
from jieba import posseg
from pysenti.compat import strdecode
from pysenti.utils import split_sentence
from pysenti import RuleClassifier
import os
import re
import pickle
import glob
import multiprocessing

from gensim.models import Word2Vec
import pandas as pd

from untils import read_events_heat, read_sample_event, combine_content, new_file_ID
from word2vec_torch import Word2VecDataset, Word2Vec_torch, train_word2vec, test_word2vec


# 覆盖 load_user_sentiment_dict 方法，完全不使用原有的 sentiment_dict
# 只使用自己的情感极性词典，不是原先的 update词典， 而是直接覆盖词典
class MyRuleClassifier(RuleClassifier):
    def load_user_sentiment_dict(self, path=None):
        if not self.inited:
            self.init()
        if path is not None:
            self.user_sentiment_dict = self._get_dict(path)
            self.sentiment_dict = self.user_sentiment_dict
        else:
            self.sentiment_dict = {}

# 测试API
def test():
    all_events = read_events_heat()
    titles = list(all_events['事件'].values)
    IDs = list(all_events['序号'].values)

    m = MyRuleClassifier()
    m.load_user_sentiment_dict()
    print("sentiment dict: ", m.sentiment_dict)

    # for title in titles:
    #     text = strdecode(title)
    #     clauses = split_sentence(text)
    #     print("clauses 分句：", clauses)
    #
    #     # 对每分句进行 分词/情感极性 分析
    #     for i in range(len(clauses)):
    #         sentence = strdecode(clauses[i])
    #         print("jieba 分词：", posseg.lcut(sentence))
    #
    #         r = m.classify(sentence)
    #         print("分句情感极性：", r['score'])
    #
    #     print('#'*20)

    for i in range(10):
        ID = IDs[i]

        # 某些文件读取失败
        try:
            sample = read_sample_event(ID)
        except:
            continue

        contents = list(sample['全文内容'].values)
        contents_short = list(sample['标题/微博内容'].values)

        # 每个事件对应几百个转发微博的全文内容
        for content in contents_short:

            # 整段文本通过API进行情感极性计算
            r = m.classify(content)
            print("文本情感极性：", r['score'])

            content = strdecode(content)
            content_clauses = split_sentence(content)

            print("clauses 分句：", content_clauses)

            # 对每 分句进行 分词
            for i in range(len(content_clauses)):
                sentence = strdecode(content_clauses[i])
                print("jieba 分词：", posseg.lcut(sentence))

            print('#'*20)


# 测试添加新词典的过程是否有效
# jieba 分词前添加新词, MyRuleClassifier 使用完全自己的词典，进行测试
# MyRuleClassifier() 中直接调用了 from jieba import posseg / posseg.lcut(sentence)，并没有独立的实例
# 因此只需要在 load_user_sentiment_dict() 前直接对 jieba.suggest_freq() 添加新词即可
def test_add_words(path):

    new_words = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            new_words.append(parts[0])

    for new_word in new_words:
        jieba.suggest_freq(new_word, tune=True)

    m = MyRuleClassifier()
    m.load_user_sentiment_dict(path)
    all_events = read_events_heat()
    titles = list(all_events['事件'].values)

    for title in titles:
        r = m.classify(title)
        print("情感极性：", r)


# 先在jieba中新增关键词，再调用 load_user_sentiment_dict() 覆盖 情感词典
def add_words(sentiment_dict_path, m):
    new_words = []
    with open(sentiment_dict_path, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            new_words.append(parts[0])

    for new_word in new_words:
        jieba.suggest_freq(new_word, tune=True)

    # MyRuleClassifier 覆盖情感词典
    m.load_user_sentiment_dict(sentiment_dict_path)


# 计算文本列表的情感极性值，并返回列表值
def cal_emotion(text_list, m):
    emotion_value = []
    try:
        for text in text_list:
            text_length = len(text)
            r = m.classify(text)
            # 情感值累计值 / 文本长度 作为文本的情感极性值
            # 情感值累计值 作为文本的情感极性值
            emotion_value.append(r['score'])
        return emotion_value
    except:
        return [0] * len(text_list)


# 计算每个事件中 '标题/微博内容' 、 '全文内容' 文本的情感极性
# 不使用已有的大词典，完全使用自己的词典进行计算
def emotion(sentiment_dict_path):
    if not os.path.exists('./data/events_emotion'):
        os.makedirs('./data/events_emotion')

    # 计算 emotion score 模型
    m = MyRuleClassifier()
    # 新增词典，覆盖情感计算值
    add_words(sentiment_dict_path, m)

    all_events = read_events_heat()
    titles = list(all_events['事件'].values)
    IDs = list(all_events['序号'].values)

    # ID对应events文件的文件名
    for i in range(len(IDs)):
        ID = IDs[i]
        title = titles[i]

        # 如果已经计算过，则跳过
        if os.path.exists('./data/events_emotion/' + str(ID) + '.xlsx'):
            continue

        # 某些文件读取失败
        try:
            # 更改后的events数据文件，可以读取
            if ID in new_file_ID:
                new_current_sample_path = os.path.join('./data/events_new', 'new' + str(ID) + '.xlsx')
                sample = pd.read_excel(new_current_sample_path)
            else:
                sample = read_sample_event(ID)
        except:
            continue

        print('当前计算事件ID: ', ID)

        contents = list(sample['全文内容'].values)
        contents_short = list(sample['标题/微博内容'].values)

        contents_score = cal_emotion(contents, m)
        contents_short_score = cal_emotion(contents_short, m)

        print('#' * 20)
        print('当前事件为：{}'.format(title))
        print("全文内容的情感平均值为: {:.2f}".format(sum(contents_score) / len(contents_score)))
        print("标题/微博内容的情感平均值为: {:.2f}".format(sum(contents_short_score) / len(contents_short_score)))

        sample['全文内容-情感极性值'] = contents_score
        sample['标题/微博内容-情感极性值'] = contents_short_score

        sample.to_excel('./data/events_emotion/' + str(ID) + '.xlsx', index=False)


# 分词、去除非中文，运算函数
def cal_train_list(data):
    stopwords_file = './data/word2vec/stopwords.txt'
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()

    train_list = []

    for text in data:
        # 使用正则表达式去除非中文字符
        try:
            text = re.sub('[^\u4e00-\u9fa5]', '', text)
            # 对文本进行分词，去除停用词
            words = posseg.cut(text)
        except:
            continue
        train_text = [word.word for word in words if word.word not in stopwords]
        train_list.append(train_text)

    return train_list

# 语料库 分词、去除非中文， 得到train_list
def split_train_list():
    file_path = './data/events/*.xlsx'
    column_name = '全文内容'

    if not os.path.exists("./data/corpus/"):
        os.makedirs("./data/corpus/")

    files = glob.glob(file_path)
    for file in files:
        file_name = os.path.basename(file)
        print(file_name)
        try:
            df = pd.read_excel(file, usecols=[column_name])
            df = df[column_name]
            train_list = cal_train_list(df)
            print("语料数量：", len(train_list))
        except:
            continue

        with open("./data/corpus/" + file_name.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(train_list, f)

    # 多个corpus数组合并为一个
    merged_list = []
    pkl_file_path = './data/corpus/*.pkl'
    train_list_corpus_path = './data/word2vec/train_list.pkl'
    files = glob.glob(pkl_file_path)
    for file in files:
        with open(file, 'rb') as f:
            array = pickle.load(f)
            merged_list.extend(array)
    # 将合并后的数组保存为新的pkl文件
    with open(train_list_corpus_path, 'wb') as f:
        pickle.dump(merged_list, f)

# 词向量模型训练
def train_w2v():
    # 训练语料库
    train_list = []
    # 判断文件是否存在
    if os.path.exists('./data/word2vec/train_list.pkl'):
        print("train_list 文件存在，直接读取")
        # 从文件中读取数组
        with open('./data/word2vec/train_list.pkl', 'rb') as f:
            train_list = pickle.load(f)
    else:
        print("train_list 文件不存在, 计算一次")
        split_train_list()

        with open('./data/word2vec/train_list.pkl', 'rb') as f:
            train_list = pickle.load(f)

    print("训练语料数量：", len(train_list))

    vector_size = 128
    window = 5
    hs = 1
    min_count = 3
    # 用CBOW模型训练词向量
    model = Word2Vec(sentences=train_list, vector_size=vector_size, sg=0, epochs=100, workers=5, window=window, min_count=min_count, hs=hs)
    # 将模型保存到磁盘
    with open('./data/word2vec/word2vec_model_CBOW.pkl', 'wb') as f:
        pickle.dump(model, f)


# 词向量模型训练 -GPU
def train_w2v_torch_gpu():
    # 训练语料库
    train_list = []
    # 判断文件是否存在
    if os.path.exists('./data/word2vec/train_list.pkl'):
        print("train_list 文件存在，直接读取")
    else:
        print("train_list 文件不存在, 计算一次")
        split_train_list()
        # # 计算核心数
        # num_cores = multiprocessing.cpu_count()
        # print('CPU 核心数：', num_cores)
        #
        # # 将数据划分为num_cores个子任务
        # corpus_split = [corpus[i::num_cores] for i in range(num_cores)]
        #
        # # 创建进程池并分配任务
        # pool = multiprocessing.Pool(num_cores)
        # results = pool.map(cal_train_list, corpus_split)
        #
        # # 合并所有子任务的结果
        # for res in results:
        #     train_list.extend(res)
        #
        # # 将数组保存到文件中
        # with open('./data/word2vec/train_list.pkl', 'wb') as f:
        #     pickle.dump(train_list, f)

    # 从文件中读取数组
    with open('./data/word2vec/train_list.pkl', 'rb') as f:
        train_list = pickle.load(f)

    print("训练语料数量：", len(train_list))

    # 定义模型参数
    embedding_dim = 128
    window_size = 1
    batch_size = 128
    num_epochs = 25
    learning_rate = 0.001

    # 定义数据集和模型
    dataset = Word2VecDataset(train_list[:1800000], window_size)
    model = Word2Vec_torch(dataset.vocab_size, embedding_dim)

    # 训练模型
    train_word2vec(model, dataset, batch_size, num_epochs, learning_rate, save_path='./data/word2vec/torch_model_word2vec.pth')

    # 测试模型
    test_word2vec(model, dataset)


# 从 original_dictionary 中读取 wordset1 的情感极性值，保存为 wordset1_ditionary.txt
def get_wordset1_value():
    original_dictionay_path = './data/word2vec/original_dictionary.txt'
    original_dictionay = {}
    with open(original_dictionay_path, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            original_dictionay[parts[0]] = parts[1]

    wordset1_path = './data/word2vec/wordset1.txt'
    wordset1_dictionay = {}
    with open(wordset1_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = str(line.strip())
            wordset1_dictionay[word] = original_dictionay[word]

    with open('./data/word2vec/wordset1_dictionary.txt', 'w', encoding='utf-8') as f:
        for word, value in wordset1_dictionay.items():
            f.write(word + ' ' + value)
            f.write('\n')


if __name__ == '__main__':
    # test()
    # test_add_words('./data/test_new_dict.txt')

    # 新闻文本情感词典地址
    sentiment_dict_path = './data/word2vec/wordset1_dictionary.txt'
    emotion(sentiment_dict_path)

    # 合并全文内容作为语料库 (没有分词)，分词在 split_train_list()中实现
    # combine_content(file_path='./data/events/*.xlsx', column_name='全文内容', save_path='./data/word2vec/contents_corpus.csv')

    # 词典操作
    # get_wordset1_value()

    # 训练 torch GPU Word2Vec 模型并保存
    # train_w2v_torch_gpu()
    # 训练 Word2Vec 模型并保存
    # train_w2v()




