from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def remove_invalid_rows(x_data):
    # 复制输入数据以保留原始数据不变
    x_data_process = x_data.copy()
    
    # 找到连续为空值或NAN的行
    is_invalid_row = x_data_process.isnull().any(axis=1)
    is_invalid_row |= x_data_process.isna().any(axis=1)
    # print("空无效值值列：",np.where(is_invalid_row[0]))
    # 找到连续13列值为0的行   注意：存在转发点赞评论为零的经过实验发现是12列最多，最后得到结果与人眼观测结果一致，得到所有去除后的数据  
    is_invalid_row |= (x_data_process == 0).astype(int).rolling(13, axis=1).sum().eq(13).any(axis=1)
    # 获取需要删除的行的索引
    invalid_row_indices = np.where(is_invalid_row)[0]
    
    # 删除无效行
    x_data_process.drop(x_data_process.index[invalid_row_indices], inplace=True)
    
    # 返回处理后的数据和删除的行索引
    return x_data_process, invalid_row_indices



def  GDBT_training(X_train, X_test, y_train, y_test,model_save_path=None):
    # 创建GradientBoostingRegressor模型
    model = GradientBoostingRegressor()

    # 定义参数搜索空间
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5]
    }

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    # 拟合训练数据
    grid_search.fit(X_train, y_train)

    # 输出最佳参数组合
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    if model_save_path != None:
        # 保存最佳模型
        joblib.dump(grid_search.best_estimator_, model_save_path)

        # 加载最佳模型
        loaded_model = joblib.load('best_model.pkl')

        # 使用加载的模型进行预测
        y_pred = loaded_model.predict(X_test)
    
        # 计算均方误差
        mse = mean_squared_error(y_test, y_pred)
        print("Mean squared error:", mse)

        return loaded_model
    return best_params


def evaluate_fuc(loaded_model,X_train, X_test, y_train, y_test):
    
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
    plt.show()