import scipy
import numpy as np

path = "/home/zengguangjie/SSSE/datasets/clustering/CovType.mat"
data_f = scipy.io.loadmat(path)

X = np.array(data_f['fea']).astype(float)

y = np.array(data_f['gnd']).astype(float).squeeze()
print(X)
print(y)

data = {}
data["fea"] = X
data['gnd'] = y

scipy.io.savemat(path, data, do_compression=True)

data_f = scipy.io.loadmat(path)

X = np.array(data_f['fea']).astype(float)

y = np.array(data_f['gnd']).astype(float).squeeze()
print(X)
print(y)

