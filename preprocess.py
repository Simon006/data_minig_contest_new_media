import pandas as pd
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from untils import read_events_heat, read_sample_event, calculate_int64_extra, calculate_days_diff, new_file_ID
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

from untils import all_columns, int64_columns, semantic_columns, emotion_columns

# 所有数据列
all_columns_ = all_columns

# 标量型数据列
int64_columns_ = int64_columns + ['时间衰减系数']

# 词频（词面量）相似性、语义相似性 数据列
semantic_columns_ = semantic_columns + ['时间衰减系数']

# 情感数据列
emotion_columns_ = emotion_columns + ['时间衰减系数']

# 标量型数据列 + 词频（词面量）相似性、语义相似性 数据列
int64_semantic_columns = int64_columns + semantic_columns + ['时间衰减系数']

# 标量型数据列 + 情感数据列
int64_emotion_columns = int64_columns + emotion_columns + ['时间衰减系数']

# 词频（词面量）相似性、语义相似性 数据列 + 情感数据列
semantic_emotion_columns = semantic_columns + emotion_columns + ['时间衰减系数']

# 聚类类别数量
n_clusters = 3


# tsne降维度后可视化并保存图像
def figure_show_save(X_tsne, labels, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot()
    cmap = plt.cm.get_cmap('jet', 3)
    # heat label 后面小热度值的全是同类别的，只取前100项看清楚点
    sc = ax.scatter(X_tsne[0:100, 0], X_tsne[0:100, 1], X_tsne[0:100, 2], c=labels[0:100], cmap=cmap)
    # sc = ax.scatter(X_tsne[0:100, 0], X_tsne[0:100, 1], c=labels[0:100], cmap=cmap)
    # 添加颜色条和标签
    cbar = plt.colorbar(sc)
    plt.savefig(save_path)
    plt.show()


# 将每个事件的热度分3类，先根据标准正态分布拟合，再聚类算法算出4类
# 直接根据热度进行聚类
# 保存新文件 './data/heat_events_withLabel.xlsx'
def calcute_heat_class():
    all_events_heat = read_events_heat()
    event_heat = all_events_heat['热度'].values.reshape(-1, 1)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    x = scaler.fit_transform(event_heat)

    # 定义聚类模型，这里将数据分为 n_clusters 类
    kmeans = KMeans(n_clusters=n_clusters, max_iter=500)

    # 训练模型并预测每个数据点所属的簇
    kmeans.fit(x)
    labels = kmeans.predict(x)
    # print(labels[340:560])

    # 将聚类结果添加到 DataFrame 中
    all_events_heat['heat_label'] = labels
    all_events_heat.to_excel('./data/heat_events_withLabel.xlsx', index=False)


# 读取每个事件的数据，将 点赞|转发|评论|粉丝数|关注数 这些标量值
# 的统计信息（平均值|最大值|方差）* 5 = 15 维度数据 算出来
# 保存到新的文件'./data/extra_1.xlsx'
err_file_ID = [104, 136, 297, 490, 496, 639, 914, 996, 1375, 1502, 1732, 1757, 1864, 2795, 2906, 3451, 3732, 3989]
def feature_int64_save(dir_allData_path='./data/events'):
    all_events_heat = read_events_heat()
    IDs = all_events_heat['序号'].values
    # 保存的list
    save_arr = []

    for ID in IDs:
        sample_event_path = os.path.join(dir_allData_path, str(ID)+'.xlsx')
        try:
            sample_event = pd.read_excel(sample_event_path)
            # 数值标量统计信息
            extra_arr = calculate_int64_extra(sample_event)
            save_arr.append([ID, *extra_arr])
        except:
            print(ID)
            save_arr.append([ID])

    save_df = pd.DataFrame(save_arr)
    save_df.columns = ['序号', '点赞-mean', '点赞-max', '点赞-var','点赞-sum',
                       '转发-mean', '转发-max', '转发-var','转发-sum',
                       '评论-mean', '评论-max', '评论-var','评论-sum',
                       '粉丝数-mean', '粉丝数-max', '粉丝数-var','粉丝数-sum',
                       '关注数-mean', '关注数-max', '关注数-var','关注数-sum']
    save_df.to_excel('./data/extra_3.xlsx', index=False)


# 将 原始 标量数据、events_semantic 和 events_emotion、时间衰减 的特征计算出来
def feature_combine_v1():
    all_events_heat = read_events_heat()
    IDs = all_events_heat['序号'].values
    # 保存的list
    save_arr = []
    for ID in IDs:
        print('事件ID: ', ID)
        child_arr = []
        # ID
        child_arr.append(ID)

        events_child_path = os.path.join('./data/events', str(ID)+'.xlsx')
        events_semantic_child_path = os.path.join('./data/events_semantic', str(ID)+'.xlsx')
        events_emotion_child_path = os.path.join('./data/events_emotion', str(ID)+'.xlsx')

        if os.path.exists(events_child_path):
            try:
                sample_event = pd.read_excel(events_child_path)
                # 数值标量统计信息
                extra_arr = calculate_int64_extra(sample_event, ['点赞', '转发', '评论', '粉丝数', '关注数'])
                child_arr.extend(extra_arr)
            except:
                child_arr.extend([np.nan] * 20)
        else:
            child_arr.extend([np.nan] * 20)

        if os.path.exists(events_semantic_child_path):
            try:
                sample_event_semantic = pd.read_excel(events_semantic_child_path)
                # 语义相关信息
                extra_arr_semantic = calculate_int64_extra(sample_event_semantic,
                ['全文内容-词频相似性', '标题/微博内容-词频相似性', '全文内容-语义相似性', '标题/微博内容-语义相似性'])
                child_arr.extend(extra_arr_semantic)
            except:
                child_arr.extend([np.nan] * 16)
        else:
            child_arr.extend([np.nan] * 16)

        if os.path.exists(events_emotion_child_path):
            try:
                sample_event_emotion = pd.read_excel(events_emotion_child_path)
                # 情感极性相关信息
                extra_arr_emotion = calculate_int64_extra(sample_event_emotion,
                ['全文内容-情感极性值', '标题/微博内容-情感极性值'])
                child_arr.extend(extra_arr_emotion)
            except:
                child_arr.extend([np.nan] * 8)
        else:
            child_arr.extend([np.nan] * 8)

        save_arr.append(child_arr)

    save_df = pd.DataFrame(save_arr)
    save_df.columns = ['序号', '点赞-mean', '点赞-max', '点赞-var', '点赞-sum',
                       '转发-mean', '转发-max', '转发-var', '转发-sum',
                       '评论-mean', '评论-max', '评论-var', '评论-sum',
                       '粉丝数-mean', '粉丝数-max', '粉丝数-var', '粉丝数-sum',
                       '关注数-mean', '关注数-max', '关注数-var', '关注数-sum',
                       '全文内容-词频相似性-mean', '全文内容-词频相似性-max', '全文内容-词频相似性-var', '全文内容-词频相似性-sum',
                       '标题/微博内容-词频相似性-mean', '标题/微博内容-词频相似性-max', '标题/微博内容-词频相似性-var', '标题/微博内容-词频相似性-sum',
                       '全文内容-语义相似性-mean', '全文内容-语义相似性-max', '全文内容-语义相似性-var', '全文内容-语义相似性-sum',
                       '标题/微博内容-语义相似性-mean', '标题/微博内容-语义相似性-max', '标题/微博内容-语义相似性-var', '标题/微博内容-语义相似性-sum',
                       '全文内容-情感极性值-mean', '全文内容-情感极性值-max', '全文内容-情感极性值-var', '全文内容-情感极性值-sum',
                       '标题/微博内容-情感极性值-mean', '标题/微博内容-情感极性值-max', '标题/微博内容-情感极性值-var', '标题/微博内容-情感极性值-sum',]

    # 时间衰减数据
    time_column_data = calculate_days_diff()["时间衰减系数"]
    save_df.insert(1, "时间衰减系数", time_column_data)

    # 保存文件
    save_df.to_excel('./data/combine_data_v1.xlsx', index=False)


# 对标量数据进行聚类         直接的数值型数据聚类，标签结果在 heat_events_Label_int64.xlsx
def KMeans_int64(n_clusters=n_clusters, save=False, cols_to_cluster=None):
    all_events_heat = read_events_heat(filename="heat_events.xlsx")
    events_extra = read_events_heat(filename="combine_data_v1.xlsx")

    # 选择要聚类的列，或者选择全部
    if cols_to_cluster is None:
        cols_to_cluster = all_columns

    df_cluster = events_extra[cols_to_cluster]
    #
    # # 使用mean()函数计算每列的平均值
    mean_values = df_cluster.mean()
    #
    # # 使用fillna()函数将缺失值替换为 平均值/零值 ？
    # df_cluster.fillna(mean_values, inplace=True)
    df_cluster.fillna(0, inplace=True)

    # 标准化
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)
    #
    kmeans = KMeans(n_clusters=n_clusters, max_iter=500)
    kmeans.fit(df_cluster_scaled)

    labels = kmeans.predict(df_cluster_scaled)
    # print(labels)

    if save == True:
        # 将聚类结果添加到 DataFrame 中
        all_events_heat['heat_label'] = labels
        all_events_heat.to_excel('./data/heat_events_Label_int64.xlsx', index=False)

    return labels


# 对标量数据进行降维，根据标签(heat标签与KMeans标签)可视化
def tSNE_int64():
    if not os.path.exists('./data/TSNE'):
        os.makedirs('./data/TSNE')

    all_events_heat = read_events_heat(filename="heat_events_withLabel.xlsx")
    labels_heat = all_events_heat['heat_label'].values.reshape(-1, 1)

    # all_events_feature_label = read_events_heat(filename="heat_events_Label_int64.xlsx")
    # labels_kmeans = all_events_feature_label['heat_label']

    cols_to_cluster = all_columns_
    pic_path1 = './data/TSNE/all_columns_heat_labels'
    pic_path2 = './data/TSNE/all_columns_kmeans_labels'

    labels_kmeans = KMeans_int64(save=True, n_clusters=n_clusters, cols_to_cluster=cols_to_cluster)

    events_extra = read_events_heat(filename="combine_data_v1.xlsx")

    df_cluster = events_extra[cols_to_cluster]
    # # 使用mean()函数计算每列的平均值
    mean_values = df_cluster.mean()
    # # 使用fillna()函数将缺失值替换为 平均值/零值 ？
    # df_cluster.fillna(mean_values, inplace=True)
    df_cluster.fillna(0, inplace=True)

    # 标准化处理
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # n_components 降维后的维度
    # tsne = TSNE(n_components=3, init='pca')
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(X_scaled)

    # 将降维后的数据集与标签一起进行可视化
    # 标签分别是：（1）热度值直接聚类的标签   （2）对数据聚类得到的标签
    figure_show_save(X_tsne, labels_heat, pic_path1)
    figure_show_save(X_tsne, labels_kmeans, pic_path2)


if __name__ == '__main__':
    calcute_heat_class()
    # feature_int64_save()
    # KMeans_int64(save=True)
    tSNE_int64()

    # 数据 combine
    # feature_combine_v1()