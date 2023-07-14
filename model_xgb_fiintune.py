import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from process_1 import load_data_from_xlrx


cols_to_cluster = ["上海","云南","其他","内蒙古","北京", "台湾","吉林","四川","天津","宁夏","安徽",	"山东","山西",
                   "广东","广西","新疆","江苏","江西","河北","河南","浙江","海南","海外","湖北","湖南","澳门","甘肃","福建","西藏","贵州",
                   "辽宁","重庆","陕西","青海","香港","黑龙江","点赞-mean","点赞-max","点赞-var","点赞-sum","转发-mean","转发-max","转发-var",
                   "转发-sum","评论-mean","评论-max","评论-var","评论-sum","粉丝数-mean","粉丝数-max","粉丝数-var","粉丝数-sum","关注数-mean",
                   "关注数-max","关注数-var","关注数-sum","时间衰减系数"]

filename = "meta_data_v1.xlsx"
X_train, X_test, y_train, y_test = load_data_from_xlrx(filename=filename,cols_to_cluster=cols_to_cluster)


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



# to do 
# 模型更多参数探索
# 模型表现提升 
# 特征重要度
# 观察不同数据列下的模型表现
# nn模型表现


# 词典 