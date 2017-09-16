
import numpy as np
import matplotlib.pyplot as pypl
import time
np.random.seed(int(time.time()))

# generate the data set
u = np.arange(0,10,0.01)
k = np.arctan(np.sin(u) + np.cos(u)) 
factor = 4 * np.max(k)
train_rand_values = np.random.rand(k.shape[0])
X = np.array(factor*(train_rand_values-np.mean(train_rand_values)))

# re-seed for croos-validation and make it more random
np.random.seed(int(time.time()))
eval_rand_values= np.random.rand() * np.random.rand(k.shape[0])
Xval = np.array(factor*(eval_rand_values-np.mean(eval_rand_values)))

# labels of X and Xval
labels = np.zeros(X.shape[0])
labels[X >= k] = 1
val_labels = np.zeros(Xval.shape[0])
val_labels[Xval >= k] = 1

ix = np.array(range(X.shape[0]))
full_X = np.column_stack((ix, X))
data = np.column_stack((full_X, labels))
val_ix = np.array(range(Xval.shape[0]))
val_full_X = np.column_stack((val_ix, Xval))
val_data = np.column_stack((val_full_X, val_labels))

X = data[:,[0,1]]
y = data[:,2].astype('int32')
Xval = val_data[:,[0,1]]
yval= val_data[:,2].astype('int32')

pos = np.array(y == 1)
neg = np.array(y == 0)
val_pos = np.array(yval == 1)
val_neg = np.array(yval == 0)

# plot the actual decision boundary
pypl.figure(num=1)
pypl.scatter(X[pos, 0], X[pos, 1], c='b', label='+ive examples')
pypl.scatter(X[neg, 0], X[neg, 1], c='r', label='-ive examples')
pypl.plot(k, c='g', label='true dec. bound.')
pypl.xlabel('Training examples feature# 1')
pypl.ylabel('Training examples feature# 2')
pypl.legend(loc='upper right')
pypl.show()

# some network parameters
m, n = X.shape
ins = 2
hid1_units = 40
hid2_units = 20
outs = 2
epsilon = 1e-3

# hidden layers initialized
Theta1 = epsilon * np.random.randn(ins,hid1_units)
b1 = np.zeros((1,hid1_units))

Theta2 = epsilon * np.random.randn(hid1_units,hid2_units)
b2 = np.zeros((1,hid2_units))

Theta3 = epsilon * np.random.randn(hid2_units,outs)
b3 = np.zeros((1,outs))

epochs = 50000
batch_size = 100
alpha = 1e-3
losses = []
e = 1

# training loop
for e in range(epochs):

    # forward pass
    A2 = np.maximum(0, X.dot(Theta1) + b1)
    A3 = np.maximum(0, A2.dot(Theta2) + b2)
    A4 = A3.dot(Theta3) + b3

    # backward pass    
    exp_scores = np.exp(A4)
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -1 / m * np.sum(np.log(prob[range(m), y]))
    
    # verbose
    if e % 500 == 0:
        print('epoch#', e, ' loss = ', loss)
    e += 1
    losses.append(loss)

    dL_dA4 = prob
    dL_dA4[range(m), y] -= 1
    dL_dA4 /= m

    dL_db3 = dL_dA4.sum(axis=0, keepdims=True)
    dL_dTh3 = np.dot(A3.T, dL_dA4)

    dL_dA3 = np.dot(dL_dA4, Theta3.T)
    dL_dA3[A3 <= 0] = 0

    dL_db2 = dL_dA3.sum(axis=0, keepdims=True)
    dL_dTh2 = np.dot(A2.T, dL_dA3)

    dL_dA2 = np.dot(dL_dA3, Theta2.T)
    dL_dA2[A2 <= 0] = 0

    dL_db1 = dL_dA2.sum(axis=0, keepdims=True)
    dL_dTh1 = np.dot(X.T, dL_dA2)

    # update synapse matrices
    Theta1 += -alpha * dL_dTh1
    b1 += -alpha * dL_db1

    Theta2 += -alpha * dL_dTh2
    b2 += -alpha * dL_db2

    Theta3 += -alpha * dL_dTh3
    b3 += -alpha * dL_db3

# check the training accuracy
train_acc = 100 * np.mean(np.argmax(A4, axis=1) == y)
print('training accuracy = ', train_acc)
# cross validation in here
A2 = np.maximum(0, Xval.dot(Theta1) + b1)
A3 = np.maximum(0, A2.dot(Theta2) + b2)
A4 = A3.dot(Theta3) + b3
val_acc = 100 * np.mean(np.argmax(A4, axis=1) == yval)
print('croos-validation accuracy = ', val_acc)

# plot the loss curve over #of iterations
pypl.figure(2)
pypl.plot(losses, c='r')
pypl.xlabel('iteration')
pypl.ylabel('cost')
pypl.show()











