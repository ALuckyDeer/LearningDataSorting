# datafram.hist()
hist()函数被定义为一种从数据集中了解某些数值变量分布的快速方法。它将数字变量中的值划分为” bins”。它计算落入每个分类箱中的检查次数。这些容器负责通过可视化容器来快速直观地了解变量中值的分布。
# df.map(function, iterable, ...)
function 可匹配lamda表达式，后面迭代器
map(labda x:x+a,数组)
# df.unique df.nunique
unique()是以 数组形式（numpy.ndarray）返回列的所有唯一值（特征的所有唯一值）

nunique() Return number of unique elements in the object.即返回的是唯一值的个数
# 在大量数据下读取数据慢解决办法
* 先把df数据转为numpy数据才索引了，数据集比较大的话这步需要耗时很久
`data.values[idx]`
* 改成直接datafram索引，快了许多倍
`data.iloc[idx,:]`
