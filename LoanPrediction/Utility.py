import numpy as np
def dict2Array(dict):
    result = []
    for key, vector in dict.items():
        result.append(vector)
    return result

a = np.array([[1, 2, 3], [4,5,6]])
b = np.array([[1, 2, 3], [4,5,6]])
print np.concatenate((a, b), axis=1)