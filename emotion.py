import jieba
from jieba import posseg
from pysenti.compat import strdecode
from pysenti.utils import split_sentence
from pysenti import RuleClassifier
import os

from untils import read_events_heat, read_sample_event


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
    for text in text_list:
        r = m.classify(text)
        emotion_value.append(r['score'])
    return emotion_value


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

        # 某些文件读取失败
        try:
            sample = read_sample_event(ID)
        except:
            continue

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



if __name__ == '__main__':
    # test()
    # test_add_words('./data/test_new_dict.txt')

    # 新闻文本情感词典地址
    sentiment_dict_path = './data/test_new_dict.txt'
    emotion(sentiment_dict_path)



