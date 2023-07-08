import pandas as pd
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from untils import read_events_heat, read_sample_event, calculate_int64_extra
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

# tsne降维度后可视化并保存图像
def figure_show_save(X_tsne, labels, save_path):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot()
    cmap = plt.cm.get_cmap('jet', 2)
    # heat label 后面小热度值的全是同类别的，只取前1000项看清楚点
    # sc = ax.scatter(X_tsne[0:30, 0], X_tsne[0:30, 1], X_tsne[0:30, 2], c=labels[0:30], cmap=cmap)
    sc = ax.scatter(X_tsne[0:100, 0], X_tsne[0:100, 1], c=labels[0:100], cmap=cmap)
    # 添加颜色条和标签
    cbar = plt.colorbar(sc)
    plt.savefig(save_path)
    plt.show()


# 将每个事件的热度分4类，先根据标准正态分布拟合，再聚类算法算出4类
# 直接根据热度进行聚类
# 保存新文件 './data/heat_events_withLabel.xlsx'
def calcute_heat_class():
    all_events_heat = read_events_heat()
    event_heat = all_events_heat['热度'].values.reshape(-1, 1)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    x = scaler.fit_transform(event_heat)

    # 定义聚类模型，这里将数据分为 3 类
    kmeans = KMeans(n_clusters=3, max_iter=500)

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


def feature_extract_v2(dir_allData_path='./data/events_semantic'):
    all_events_heat = read_events_heat()
    IDs = all_events_heat['序号'].values
    # 保存的list
    save_arr = []
    for ID in IDs:
        sample_event_path = os.path.join(dir_allData_path, str(ID)+'.xlsx')
        # 数值标量统计信息
        try:
            sample_event = pd.read_excel(sample_event_path)
            # 数值标量统计信息
            extra_arr = calculate_int64_extra(sample_event)
            save_arr.append([ID, *extra_arr,])
        except:
            print(ID)
            save_arr.append([ID])



# 对标量数据进行聚类         直接的数值型数据聚类，标签结果在 heat_events_Label_int64.xlsx
def KMeans_int64(n_clusters=2, save=False, cols_to_cluster=None):
    all_events_heat = read_events_heat(filename="heat_events.xlsx")
    events_extra = read_events_heat(filename="extra_2.xlsx")

    # 选择要聚类的列，或者选择全部
    if cols_to_cluster is None:
        cols_to_cluster = ['点赞-mean', '点赞-max', '点赞-var',
                           '转发-mean', '转发-max', '转发-var',
                           '评论-mean', '评论-max', '评论-var',
                           '粉丝数-mean', '粉丝数-max', '粉丝数-var',
                           '关注数-mean', '关注数-max', '关注数-var']

    df_cluster = events_extra[cols_to_cluster]
    #
    # # 使用mean()函数计算每列的平均值
    mean_values = df_cluster.mean()
    #
    # # 使用fillna()函数将缺失值替换为平均值
    df_cluster.fillna(mean_values, inplace=True)

    # 标准化
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)
    #
    kmeans = KMeans(n_clusters=n_clusters, max_iter=500)
    kmeans.fit(df_cluster_scaled)

    labels = kmeans.predict(df_cluster_scaled)
    print(labels)

    if save == True:
        # 将聚类结果添加到 DataFrame 中
        all_events_heat['heat_label'] = labels
        all_events_heat.to_excel('./data/heat_events_Label_int64.xlsx', index=False)

    return labels


# 对标量数据进行降维，根据标签(heat标签与KMeans标签)可视化
def tSNE_int64():
    all_events_heat = read_events_heat(filename="heat_events_withLabel.xlsx")
    labels_heat = all_events_heat['heat_label'].values.reshape(-1, 1)

    # all_events_feature_label = read_events_heat(filename="heat_events_Label_int64.xlsx")
    # labels_kmeans = all_events_feature_label['heat_label']

    cols_to_cluster = ['点赞-mean',
                       '转发-mean',
                       '评论-mean',
                       '粉丝数-mean',
                       '关注数-mean']

    labels_kmeans = KMeans_int64(save=True, n_clusters=2, cols_to_cluster=cols_to_cluster)

    events_extra = read_events_heat(filename="extra_1.xlsx")

    df_cluster = events_extra[cols_to_cluster]
    # # 使用mean()函数计算每列的平均值
    mean_values = df_cluster.mean()
    # # 使用fillna()函数将缺失值替换为平均值
    df_cluster.fillna(mean_values, inplace=True)

    # 标准化处理
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # n_components 降维后的维度
    tsne = TSNE(n_components=2, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)

    # 将降维后的数据集与标签一起进行可视化
    # 标签分别是：（1）热度值直接聚类的标签   （2）对数值型数据聚类得到的标签
    figure_show_save(X_tsne, labels_heat, './data/heat_labels_tsne')
    figure_show_save(X_tsne, labels_kmeans, './data/kmeans_labels_tsne')





if __name__ == '__main__':
    # calcute_heat_class()
    feature_int64_save()
    # KMeans_int64(save=True)
    # tSNE_int64()