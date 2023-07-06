import pandas as pd
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from untils import read_events_heat, read_sample_event
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge,LinearRegression
from untils import read_events_heat, read_sample_event, clacute_int64_extra
import xgboost as xgb
from sklearn.metrics import r2_score#R square
from sklearn.model_selection import cross_val_score
# 简单查看所有数据情况
def look_all_data(dir_path='./data', dir_allData_path='./data/events'):
    events_heat_path = os.path.join(dir_path, "heat_events.xlsx")  # 所有时间热度的excel
    files = os.listdir(dir_allData_path)                           # 具体事件数据文件

    sample_event_path = os.path.join(dir_allData_path, files[1])   # 某个具体事件数据地址

    all_events_heat = pd.read_excel(events_heat_path)
    sample_event = pd.read_excel(sample_event_path)
    pd.set_option('display.max_columns', None)
    data_all_events_heat = all_events_heat.head()                # 事件热度数据
    data_sample_event = sample_event.head()                      # 具体事件的数据，1000条

    # print(data_all_events_heat)
    # print(data_sample_event)

    # 序号|事件|热度|起始时间
    print(all_events_heat.info())                               # 看所有事件热度有些什么列
    # 检索ID|标题/微博内容|全文内容|点赞|转发|评论|账号昵称UID加密|粉丝数|关注数|地域
    print(sample_event.info())                                  # 看某个事件数据有些什么列


# 根据数值型数据进行回归拟合
def CART_fit():
    all_events_heat = read_events_heat(filename="heat_events.xlsx")
    heats = all_events_heat['热度'].values
    y = np.array(heats)

    events_extra = read_events_heat(filename="extra_1.xlsx")
    cols_to_cluster = ['点赞-mean', '点赞-max', '点赞-var',
                       '转发-mean', '转发-max', '转发-var',
                       '评论-mean', '评论-max', '评论-var',
                       '粉丝数-mean', '粉丝数-max', '粉丝数-var',
                       '关注数-mean', '关注数-max', '关注数-var']

    df_cluster = events_extra[cols_to_cluster]
    mean_values = df_cluster.mean()
    # # 使用fillna()函数将缺失值替换为平均值
    df_cluster.fillna(mean_values, inplace=True)

    X = np.array(df_cluster.values)

    # 标准化处理
    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(df_cluster)
    # X = X_scaled

    # 创建一个回归树模型
    model = DecisionTreeRegressor(max_depth=6)
    # 使用数据拟合模型
    model.fit(X, y)
    # 对训练数据进行预测
    y_pred = model.predict(X)

    # 绘制拟合曲线
    plt.plot(range(len(y)), y, 'o', label='data')
    plt.plot(range(len(y)), y_pred, label='prediction')
    plt.xlabel('index')
    plt.ylabel('regression value')
    plt.title('CART Regression')
    plt.legend()
    plt.savefig('./data/CART/regression')
    plt.show()

    # 可视化模型
    # 绘制图像
    # 指定图幅大小
    plt.figure(figsize=(18, 12))
    _ = tree.plot_tree(model, filled=True, feature_names=cols_to_cluster)
    plt.savefig('./data/CART/graph.jpg')
    plt.show()

def load_data_from_xlrx(filename=None ,cols_to_cluster=None):
    all_events_heat = read_events_heat(filename="heat_events.xlsx")
    heats = all_events_heat['热度'].values
    y = np.array(heats)

    if filename == None:
        filename = "extra_3.xlsx"
    if cols_to_cluster == None:
        cols_to_cluster = ['点赞-mean', '点赞-max', '点赞-var',
                       '转发-mean', '转发-max', '转发-var',
                       '评论-mean', '评论-max', '评论-var',
                       '粉丝数-mean', '粉丝数-max', '粉丝数-var',
                       '关注数-mean', '关注数-max', '关注数-var']

    events_extra = read_events_heat(filename=filename)
    df_cluster = events_extra[cols_to_cluster]
    mean_values = df_cluster.mean()
    # # 使用fillna()函数将缺失值替换为平均值
    df_cluster.fillna(mean_values, inplace=True)
    X = np.array(df_cluster.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def test_DecisionTreeRegressor_depth(*data,maxdepth,save_fig=False):
    X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_scores=[]
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
 
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="traing score")
    ax.plot(depths,testing_scores,label="testing score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    
    if save_fig:
        plt.savefig(f"./data/decision tree depth exploration/depth_search_graph_extract_3.jpg")
    plt.show()



def test_multiple_model_training(x_train,y_train,x_valid,y_valid):

    # 构建几个回归模型
    lr = LinearRegression()
    ridge = Ridge()
    dtr = DecisionTreeRegressor()
    xgbr = xgb.XGBRegressor()
    # 不调参看一下效果
    lr.fit(x_train, y_train)
    pred = lr.predict(x_valid)
    print("MSE:{} in LR".format(mean_squared_error(y_valid, pred)))
    print("R2_score:{} in LR".format(r2_score(y_valid, pred)))


    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_valid)
    print("MSE:{} in Ridge".format(mean_squared_error(y_valid, pred)))
    print("R2_score:{} in Ridge".format(r2_score(y_valid, pred)))


    dtr.fit(x_train, y_train)
    pred = dtr.predict(x_valid)
    print("MSE:{} in DTR".format(mean_squared_error(y_valid, pred)))
    print("R2_score:{} in DTR".format(r2_score(y_valid, pred)))

    xgbr.fit(x_train, y_train)
    pred = xgbr.predict(x_valid)
    print("MSE:{} in XGBR".format(mean_squared_error(y_valid, pred)))
    print("R2_score:{} in XGBR".format(r2_score(y_valid, pred)))


def XGBoost_fintune(X_train,X_test,Y_train,Y_test,mode,visualize=False):
    if mode == "coarse":
        print("\nstart coarse tuning:")
        ax_n_estimators = range(10,500,10)
        result = []
        for i in ax_n_estimators:
            reg = xgb.XGBRegressor(n_estimators = i,random_state = 100)
            result.append(cross_val_score(reg,X_train,Y_train,cv=5).mean())
        if visualize:
            plt.figure(figsize=(20,5))
            plt.plot(ax_n_estimators,result,c="red",label="XGB")
            plt.legend("粗调")
            plt.show()
        print("success")    

    elif mode == "finetune":
        # 精调前按照粗调修改范围
        ax_n_estimators = range(10,500,10)
        result = []
        for i in ax_n_estimators:
            reg = xgb.XGBRegressor(n_estimators = i,random_state = 100)
            result.append(cross_val_score(reg,X_train,Y_train,cv=5).mean())
        if visualize:
            plt.figure(figsize=(20,5))
            plt.plot(ax_n_estimators,result,c="red",label="XGB")
            plt.legend("细调")
            plt.show()
        # 打印最大result值，及其位置print(axisx[result.index(max(result))],max(result))
        
    
    elif mode == "booster":
        for booster in ["gbtree","gblinear","dart"]:
            reg = xgb.XGBRegressor(
                n_estimators = 180,
                learning_rate = 0.1,
                random_state = 420,
                booster = booster,
                verbosity= 2
            )
            reg.fit(X_train, Y_train)
            print(booster,":",reg.score(X_test,Y_test))
        return reg 
    

    else:
        print("false mode")
        return
    return 




if __name__ == '__main__':
    # look_all_data()
    # CART_fit()
    
    X_train,X_test,y_train,y_test = load_data_from_xlrx()
    # test_DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test,maxdepth=12,save_fig=True)
    # test_multiple_model_training(X_train,y_train,X_test,y_test)
    XGBoost_fintune(X_train,X_test,y_train,y_test,mode="coarse",visualize=True)