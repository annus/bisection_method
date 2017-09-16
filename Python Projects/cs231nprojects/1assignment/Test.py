import numpy as np
# import tensorflow as tensor

X = np.array([[1,2,-3],[4,-5,0],[-7,8,9]])
Y = X.clip(min=0,max=1)
print(Y)

#print(X)
#print(X)


