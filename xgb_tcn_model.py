
'''
 XGB-TCN model based on dataset

 2021/12 organized by LIU
'''

import pandas as pd
import numpy as np
from copy import copy, deepcopy
from numpy import concatenate
from pandas import DataFrame, concat
import os
import math
from math import sqrt
import random
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, \
    Dropout, LeakyReLU, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.layers import RepeatVector, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import GridSearchCV
import scipy
import joblib
import hyperopt
from hyperopt import fmin, tpe, hp, partial, STATUS_OK, Trials
import warnings
warnings.filterwarnings("ignore")
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置服务器GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 设置每个GPU应该拿出多少容量给进程使用，0.8代表 40%
session = InteractiveSession(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
gpus = tf.config.list_physical_devices('GPU')
print("gpu:", gpus)


MODEL_STYLE = 1
time_steps = 64
pre_len = 16
features_dim = 3

# 设置种子，为了可复现
seed = 5
np.random.seed(seed)

def subtime(date1, date2):
    date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1

value = pd.read_csv('xgb3_sc_trainX.csv') # 训练集X（归一化后）
Y = pd.read_csv('xgb3_sc_trainY.csv') # 训练集标签Y（归一化后）
testX = pd.read_csv('xgb3_sc_testX.csv') # 测试集X（归一化后）
testY = pd.read_csv('xgb3_sc_testY.csv') # 测试集标签Y（归一化后）
y_base = pd.read_csv('xgb3_sc_base.csv')  # 测试集的工程预测值（归一化后）
ori_v = pd.read_csv('xgb3_ori_v.csv')  # 测试集的实况功率值（实际值）
base_v = pd.read_csv('xgb3_base_v.csv')  # 测试集的工程预测值（实际值）

# Get data
value = (value.values).astype('float32')
print("value.shape = ", value.shape)
Y = (Y.values).astype('float32')
print("Y.shape = ", Y.shape)

train = value.reshape((value.shape[0], time_steps, features_dim))
train_x = value.flatten()
train_x.resize(train.shape[0], time_steps, features_dim)
train_y = Y

# creat test set
testX = (testX.values).astype('float32')
print("testX.shape = ", testX.shape)

testY = (testY.values).astype('float32')
print("testY.shape = ", testY.shape)

y_base = (y_base.values).astype('float32')
ori_v = (ori_v.values).astype('float32')
base_v = (base_v.values).astype('float32')

x_test = testX.reshape((testX.shape[0], time_steps, features_dim))
y_test = testY.reshape((testY.shape[0], pre_len))
print("test:", x_test.shape, y_test.shape, y_base.shape, ori_v.shape)
test_day = x_test.shape[0]

# 下载归一化的fit参数
sc = joblib.load(r'power_sc.pkl')

# 导入数据集
print("Input Data and Shape: ", train_x.shape, x_test.shape)

# design network
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, mode='min')
my_callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, verbose=2, mode='min'),
    ReduceLROnPlateau(monitor='loss', patience=5, mode='min')]
# init = tf.keras.initializers.glorot_uniform(seed=None)
init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)

#  Build TCN
def ResBlock(x, n_filters, kernel_size, dilation_rate):
    r = Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_initializer=init)(
        x)  # 第一卷积
#    r = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)(r)  # MSRA初始化
    r = LeakyReLU(alpha=0.04)(r)
    r = Dropout(0.2)(r)
    r = Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_initializer=init)(
        r)  # 第二卷积
    #    r = WeightNormalization(axis=1, momentum=0.99, epsilon=0.001)(r)
    r = LeakyReLU(alpha=0.04)(r)
    r = Dropout(0.2)(r)
    if x.shape[-1] == n_filters:
        shortcut = x
    else:
        shortcut = Conv1D(n_filters, kernel_size=1, padding='causal')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)
    return o

def TCN(train_x, train_y):
    inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    x = ResBlock(inputs, n_filters=16, kernel_size=1, dilation_rate=1)
#    x = Dropout(0.3)(x)
    x = ResBlock(x, n_filters=180, kernel_size=1, dilation_rate=2)
#    x = Dropout(0.2)(x)
    # x = ResBlock(x, n_filters=16, kernel_size=3, dilation_rate=4)
#    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(80, activation='relu')(x)
    outputs = Dense(pre_len, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    # 编译模型
    Nadam = optimizers.Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001)
    model.compile(loss="mean_squared_error", optimizer=Nadam)
    # 查看网络结构
    model.summary()
#    print(model.name)
    startdate = datetime.now()  # 获取当前时间
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式
    train_history = model.fit(train_x, train_y, epochs=200, batch_size=100, callbacks=my_callbacks,
                              validation_split=0.2, verbose=2, shuffle=True)
    enddate = datetime.now()
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式
    # 计算训练时长
    print('start date ', startdate)
    print('end date ', enddate)
    print('Time ', subtime(startdate, enddate))  # enddate > startdate

    plt.plot(train_history.history['loss'], label='train')
    plt.plot(train_history.history['val_loss'], label='valid')
    plt.legend()
    plt.show()

    return model


if MODEL_STYLE == 0:
    model = load_model(r'xgb_tcn.h5')
elif MODEL_STYLE == 1:
    model = TCN(train_x, train_y)


#  整个Test预测
pred = []
ori = []

y_rmse = []
y_base_rmse = []
y_rmse_v = []
y_base_rmse_v = []

def compute_rmse(predict, true, num):
    rmse = sqrt(mean_squared_error(true, predict))
    print('The RMSE of Test %d = %.4f' % (num, rmse))
    return rmse

for n in range(test_day):
    print("第%i次预测: " % (n + 1))
    test_x = x_test[n].reshape(1, time_steps, features_dim)
    test_y = y_test[n, :]
    predictation = model.predict(test_x)
    predict = predictation[0, :]
    true = test_y.reshape(pre_len, )
    pred.extend(predict)
    ori.extend(true)
    compute_rmse(predict, true, n + 1)

pre = np.array(pred).reshape(test_day, pre_len)
ori = np.array(ori).reshape(test_day, pre_len)
pre_v = sc.inverse_transform(pre)   # 将模型的预测值反归一化
print("pre_v:", pre_v.shape)
print("ori_v:", ori_v.shape)
print("base:", base_v.shape, y_base.shape)

for i in range(test_day):
    y_rmse.append(sqrt(mean_squared_error(ori[i], pre[i])))
    y_rmse_v.append(sqrt(mean_squared_error(ori_v[i], pre_v[i])))
    y_base_rmse_v.append(sqrt(mean_squared_error(ori_v[i], base_v[i])))
    y_base_rmse.append(sqrt(mean_squared_error(ori[i], y_base[i])))

rmse = np.mean(y_rmse)
rmse_v = np.mean(y_rmse_v)
rmse_base = np.mean(y_base_rmse)
rmse_base_v = np.mean(y_base_rmse_v)
print("rmse=%.4f" % rmse)
print("rmse_v=%.2f" % rmse_v)
print("rmse_base=%.4f" % rmse_base)
print("rmse_base_v=%.2f" % rmse_base_v, '\n')


whether = input("Enter your input: ")
data = pd.DataFrame(pre_v)
if whether == '1':
    model.save(r'xgb3_sc_tcn.h5')
    data.to_csv(r'xgb3_sc_tcn_pre.csv', index=False)
    print("XGB3-TCN MODEL Saved!")
elif whether == '0':
    print("Over!")

day = []
for n in range(test_day):
    #   n = np.random.randint(0, 31)
    print("Day%d:" % (n + 1))
    print("实况功率值: ", ori_v[n])
    print("模型预测值：", pre_v[n], "rmse=%.2f" % y_rmse_v[n])
    print("所给预测值：", base_v[n], "rmse=%.2f" % y_base_rmse_v[n], '\n')
    day.append(n)


# plot

font1 = {'family': 'Times New Roman', 'size': 24}

def plot_img(train, ori, tcn, base, day, tcn_rmse, base_rmse):
    plt.figure(figsize=(10, 8))
    plt.title("RMSE(MW) of Test %i:   XGB-TCN=%.2f,   BASE=%.2f" %
              (day+1, tcn_rmse/1000, base_rmse/1000), font1)
    tcn_pre = np.append(train[-1], tcn)
    base_pre = np.append(train[-1], base)
    ori = np.append(train[-1], ori)

    plt.plot([x for x in train], c='black')
    train = train[:-1]
#    plt.plot(test, c='black', label='True', linewidth=2)
    plt.plot([x for x in train], c='black')                  # 预测时刻前的实况功率
    plt.plot([None for _ in train] + [x for x in ori], c='black', label='TRUE', linewidth=2.5, markersize=12)  # 真实值
    plt.plot([None for _ in train] + [x for x in base_pre], c='hotpink', label='BASE', linewidth=2.5, markersize=12)  # 工程预测值
    plt.plot([None for _ in train] + [x for x in tcn_pre], c='m', label='XGB-TCN', linewidth=2.5, markersize=12) # 模型预测值

    plt.xticks(np.arange(0, 32, 4), fontsize=20)
    plt.yticks(fontsize=20)
    ymax = max(max(train), max(base), max(tcn), max(ori)) * 1.1
    ymin = min(min(train), min(base), min(tcn), min(ori)) - 800
    plt.ylim(ymin, ymax)
    plt.vlines(x=15, ymin=ymin, ymax=ori[0], colors='black', linestyles="dashed", linewidth=2)
    #    plt.axis('tight')

    plt.xlabel('timesteps / 15min', fontsize=22)
    plt.ylabel('POWER/MW', fontsize=22, rotation=360, loc='upper right')
    plt.legend(loc='best', fontsize=22)
#   plt.show()

for n in range(test_day):
    plot_img(x_test[n, 48:, -1], ori_v[n], pre_v[n], base_v[n], n, y_rmse_v[n], y_base_rmse_v[n])
plt.show()

