# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:03:59 2019

@author: hardyliu
"""

import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.arima_model import ARMA
from datetime import datetime

#加载数据
df = pd.read_csv('bitcoin_2012-01-01_to_2018-10-31.csv')

#预览数据
print(df.describe())

#将Timestamp转为时间，这一步很重要
df.Timestamp = pd.to_datetime(df.Timestamp)

#将该字段作为df的索引
df.index= df.Timestamp

#按月，季度，年来统计
df_month = df.resample('M').mean()

df_Q = df.resample('Q-DEC').mean()

df_year = df.resample('A-DEC').mean()


fig = plt.figure(figsize=(15,7))

plt.rcParams['font.sans-serif']=['SimHei']

plt.suptitle('比特币金额(美金)',fontsize=20)

plt.subplot(221)

plt.plot(df.Weighted_Price,'-',label='按天')

plt.legend()

plt.subplot(222)

plt.plot(df_month.Weighted_Price,'-',label='按月')

plt.legend()

plt.subplot(223)

plt.plot(df_Q.Weighted_Price,'-',label='按季度')

plt.legend()

plt.subplot(224)

plt.plot(df_year.Weighted_Price,'-',label='按年')
plt.legend()


plt.show()

#设置ARMA模型的参数选择范围
ps = range(0,3)
qs = range(0,3)

parameters = product(ps,qs)

prameters_list = list(parameters)

results = []

best_aic=float('inf')

#计算ARMA时间序列模型的最优解
for param in prameters_list:
    try:
        model = ARMA(df_month.Weighted_Price,order=(param[0],param[1])).fit()
    except ValueError:
        print('param error',param)
        continue
    
    aic = model.aic
    if aic<best_aic:
        best_model=model
        best_aic=aic
        best_param =param
        
    results.append([param,model.aic])   
            

result_table=pd.DataFrame(results)
result_table.columns=['parameters','aic']

print('最优模型:',best_model.summary())

df_month2 = df_month[['Weighted_Price']]

#预测的时间范围
date_list=[datetime(2018,11,30),datetime(2018,12,31),datetime(2019,1,31),
           datetime(2019,2,28),datetime(2019,3,31),
           datetime(2019,4,30),datetime(2019,5,31),datetime(2019,6,30)]


future = pd.DataFrame(index=date_list,columns=df_month.columns)

df_month2 = pd.concat([df_month2,future])
#定义保存预测结果的新列
df_month2['forecast']=best_model.predict(start=0,end=91)

plt.figure(figsize=(20,7))

df_month2.Weighted_Price.plot(label='实际金额')
df_month2.forecast.plot(color='r',ls='--',label='预测金额')
plt.legend()

plt.title('比特币金额（月）')
plt.xlabel('时间')
plt.ylabel('美金')

plt.show()