# coding:utf-8

# from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
import joblib

def dump_load():
    """
    模型保存和加载
    :return:None
    """
    # 1.获取数据
    # boston = load_boston()
    # print(boston)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 2.数据基本处理
    # 2.1 分割数据
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22, test_size=0.2)
    # 3.特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习-线性回归
    # # 4.1 模型训练
    # # estimator = Ridge(alpha=1.0)
    # estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
    # estimator.fit(x_train, y_train)
    #
    # # 4.2 模型保存
    # joblib.dump(estimator, "./data/test.pkl")

    # 4.3 模型训练
    estimator = joblib.load("./data/test.pkl")

    print("这个模型的偏置是:\n", estimator.intercept_)
    print("这个模型的系数是:\n", estimator.coef_)

    # 5.模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("预测值是:\n", y_pre)

    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差:\n", ret)





if __name__ == '__main__':
    dump_load()
