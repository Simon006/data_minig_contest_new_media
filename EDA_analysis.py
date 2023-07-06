import pandas as pd
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import xlrd
dir_1_path = "D:\新传数据挖掘比赛\数据集\数据集\选题二训练集：热点事件的发展趋势预测\数据"
files = os.listdir(dir_1_path)
# print(files)

print(os.path.join(dir_1_path,files[0]))
#print(os.join(os.getcwd))
sample_1 = os.path.join(dir_1_path,files[0])



wb_1 = xlrd.open_workbook(sample_1)
sh = wb_1.sheet_by_name("1")
print(sh.nrows)#有效数据行数     # 行数包括了行头 输出1001 实际1000条数据
print(sh.ncols)#有效数据列数
print(sh.cell(0,0).value)#输出第一行第一列的值
print(sh.row_values(0))#输出第一行的所有值
#将数据和标题组合成字典
print(dict(zip(sh.row_values(0),sh.row_values(1))))
#遍历excel，打印所有数据
# for i in range(sh.nrows):
for i in range(10):
    print(sh.row_values(i))





# 用pd读取
df_sample_2 = pd.read_excel(sample_1)
data=df_sample_2.values
# print("获取到所有的值:\n{}".format(data))
print(type(data))
print(data.shape)


import pandas as pd
import re
 
#创建DataFrame
df1 = pd.DataFrame([['2015-03-24'],['2011-07-12'],['2010-02-08']])
 
#使用apply()和lambda进行提取月份
df1 = df1[0].apply(lambda x:re.findall('\d+-(\d+)-\d+',x)[0])

