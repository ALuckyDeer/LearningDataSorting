import numpy as np
from sklearn.model_selection import KFold

a = np.arange(27).reshape(9, 3)
print(a)
b = np.arange(9).reshape(9, 1)
kfold = KFold(n_splits=3, shuffle=False)
#index = kfold.split(X=a)
#print(list(index))
#print(type(index))
index = kfold.split(X=a, y=b)
#print(list(index))
for train_index, test_index in index:
    print("-------------------------------------------------")
    print(a[train_index],b[train_index]) #注意如果a是datafram类型就得用a.iloc[tain_index], 因为a[train_index]会被认为是访问列
    print(a[test_index],b[test_index])