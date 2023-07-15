import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
from process_1 import load_data_from_xlrx
from matplotlib import pyplot as plt
import joblib
import time
# cols_to_cluster = ["上海","云南","其他","内蒙古","北京", "台湾","吉林","四川","天津","宁夏","安徽",	"山东","山西",
#                    "广东","广西","新疆","江苏","江西","河北","河南","浙江","海南","海外","湖北","湖南","澳门","甘肃","福建","西藏","贵州",
#                    "辽宁","重庆","陕西","青海","香港","黑龙江","点赞-mean","点赞-max","点赞-var","点赞-sum","转发-mean","转发-max","转发-var",
#                    "转发-sum","评论-mean","评论-max","评论-var","评论-sum","粉丝数-mean","粉丝数-max","粉丝数-var","粉丝数-sum","关注数-mean",
#                    "关注数-max","关注数-var","关注数-sum","时间衰减系数"]


cols_to_cluster = ["时间衰减系数","点赞-mean","点赞-max","点赞-var","点赞-sum","转发-mean","转发-max","转发-var","转发-sum","评论-mean",
                   "评论-max","评论-var","评论-sum","粉丝数-mean","粉丝数-max","粉丝数-var","粉丝数-sum",
                   "关注数-mean","关注数-max","关注数-var","关注数-sum","全文内容-词频相似性-mean",
                   "全文内容-词频相似性-max","全文内容-词频相似性-var","全文内容-词频相似性-sum","标题/微博内容-词频相似性-mean",
                   "标题/微博内容-词频相似性-max","标题/微博内容-词频相似性-var","标题/微博内容-词频相似性-sum","全文内容-语义相似性-mean",
                   "全文内容-语义相似性-max","全文内容-语义相似性-var","全文内容-语义相似性-sum","标题/微博内容-语义相似性-mean",
                   "标题/微博内容-语义相似性-max","标题/微博内容-语义相似性-var","标题/微博内容-语义相似性-sum",
                   "全文内容-情感极性值-mean","全文内容-情感极性值-max","全文内容-情感极性值-var","全文内容-情感极性值-sum",
                   "标题/微博内容-情感极性值-mean","标题/微博内容-情感极性值-max","标题/微博内容-情感极性值-var","标题/微博内容-情感极性值-sum",
                   "一次转发","二次转发","三次转发","四次转发","大于四次转发","总转发次数","大于等于四次转发","得分","归1化","上海",
                   "云南","其他","内蒙古","北京","台湾","吉林","四川","天津","宁夏","安徽","山东","山西","广东","广西","新疆","江苏","江西","河北",
                   "河南","浙江","海南","海外","湖北","湖南","澳门","甘肃","福建","西藏","贵州","辽宁","重庆","陕西","青海","香港","黑龙江"]




def xgb_finetune_v1(X_train, y_train):
    cv_params = {'n_estimators': [400, 500, 600, 700, 800],'learning_rate': [0.01 ,0.015, 0.1, 0.005, 0.0001],'reg_alpha': [0, 0.1, 0.01]}
    other_params = { 'n_estimators': 500, 'max_depth': 8, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,  'reg_lambda': 1}



    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    # 参数的最佳取值：{'learning_rate': 0.0001, 'n_estimators': 800, 'reg_alpha': 0}
    # 最佳模型得分:0.026395031769172638




def xgb_best_version_visualize(X_train, X_test, y_train, y_test,cols_to_cluster,save=False):
    # 中文显示
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 参数的最佳取值：{'learning_rate': 0.005, 'n_estimators': 400, 'reg_alpha': 0.1}
    params = {'learning_rate': 0.005, 'n_estimators': 400, 'reg_alpha': 0.1, 'max_depth': 8, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,  'reg_lambda': 1}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train,y_train)

    
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    if save == True:
        # save model to file
        joblib.dump(model, f"./data/xgb_model_train_{train_acc}_test_{test_acc}.xgb")
    print("train accuracy:",train_acc)
    print("test accuracy:",test_acc)
    importance = model.feature_importances_
    Impt_Series = pd.Series(importance, index = cols_to_cluster)
    print(Impt_Series)



    # Impt_Series.sort_values(ascending = True).plot('barh')
    Impt_Series = Impt_Series.sort_values(ascending = True)
    Impt_Series = Impt_Series[50:]
    print(Impt_Series)
    print(list(Impt_Series.index))
    Y = list(Impt_Series.index)
    # 绘制条形图
    plt.figure(figsize=(30,15)) 
    plt.barh(range(len(Y)), # 指定条形图y轴的刻度值
            Impt_Series.values, # 指定条形图x轴的数值
            tick_label = Y, # 指定条形图y轴的刻度标签
            color = 'steelblue', # 指定条形图的填充色
        )

    print(Impt_Series.values)
    # print()
    for y,x in enumerate(Impt_Series.values):
        plt.text(x+0.0001,y,'%s' %round(x,3),va='center')

    plt.savefig("./data/XGBoost_feature_importance.jpg")    
    plt.show()
    return model

def viusalize_residual(loaded_model,X_train, X_test, y_train, y_test):
    # 在训练集和测试集上进行预测
    y_train_pred = loaded_model.predict(X_train)
    y_test_pred = loaded_model.predict(X_test)

    # 计算预测与真实值之间的残差
    train_residuals = y_train_pred - y_train
    test_residuals = y_test_pred - y_test

    # 绘制残差图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_pred, train_residuals, c='blue', label='Training Set')
    plt.scatter(y_test_pred, test_residuals, c='red', label='Test Set')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.savefig('./data/Residual_Plot_all_data.jpg')
    plt.show()

def load_xgb_model(model_path):
    loaded_model = joblib.load(model_path)
    return loaded_model

if __name__ == '__main__':
    filename = "combine_data_v3_无序号.xlsx"
    X_train, X_test, y_train, y_test = load_data_from_xlrx(filename=filename,cols_to_cluster=cols_to_cluster)
    # # xgb_finetune_v1(X_train, y_train)
    # model = xgb_best_version_visualize(X_train, X_test, y_train, y_test,cols_to_cluster,save=True)
    # viusalize_residual(model,X_train, X_test, y_train, y_test)
    path2 = r"D:\新传数据挖掘比赛\Code\Contest\data_minig_contest_new_media\data\xgb_model_train_0.8874434136105651_test_0.6272666879929087.pkl"
    loaded_model = load_xgb_model(path2)
    # loaded_model visualize
    viusalize_residual(loaded_model,X_train, X_test, y_train, y_test)