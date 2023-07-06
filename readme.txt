/data
    /CART                               决策树输出结果
    /events                             原始每个事件的详细数据
    /events_semantic                    每个事件的 '标题/微博内容' 、 '全文内容' 和 '事件' 的 语义相似度 \ Tfidf
    heat_events.xlsx                    原始每个事件热度等数据
    heat_events_withLabel.xlsx          根据原始事件热度简单聚类，得到的聚类标签
    heat_events_Label_int64.xlsx        根据每个事件的数值型数据进行聚类，得到的聚类标签


utils.py
            常用的一些函数

preprocess.py
      def feature_int64_save():          数值型数据预处理，
                                         每个事件的转发微博的 ['点赞', '转发', '评论', '粉丝数', '关注数']
                                         计算 平均值/最大值/方差 等等
      def KMeans_int64():                根据数值型数据进行聚类，得到数值型数据标签：heat_events_Label_int64.xlsx
      def calcute_heat_class():          根据每个事件的热度进行聚类，得到热度标签：heat_events_withLabel.xlsx
      def tSNE_int64():                  对数值型数据进行降维可视化，并且与 数值型数据标签/热度标签 进行展示

process_1.py
      def look_all_data():               简单查看数据情况
      def CART_fit():                    根据数值型数据向热度值进行回归拟合
      def XGBoost_fintune():             用于XGB模型精调的函数
      def test_multiple_model_training():用于多模型测试
       

semantic.py
       def cal_semantic():               根据 'heat_events' 中 '标题/微博内容' 、 '全文内容' 和 '事件'
                                         计算语义相似度 \ Tfidf
                                         得到的特征保存在 /events_semantic 中

       计算语义相似度用到的中文NLP模型：      https://github.com/shibing624/text2vec
                                         https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase