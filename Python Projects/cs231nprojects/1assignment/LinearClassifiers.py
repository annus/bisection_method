#  this whole part is needed to load the data from the CIFAR10 dataset
from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import matplotlib.pyplot as pypl
print('All Packages imported!')


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'Cifar10data'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
      mean_image = np.mean(X_train, axis=0)
      X_train -= mean_image
      X_val -= mean_image
      X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
    

def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
  """
  Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
  TinyImageNet-200 have the same directory structure, so this can be used
  to load any of them.

  Inputs:
  - path: String giving path to the directory to load.
  - dtype: numpy datatype used to load the data.
  - subtract_mean: Whether to subtract the mean training image.

  Returns: A dictionary with the following entries:
  - class_names: A list where class_names[i] is a list of strings giving the
    WordNet names for class i in the loaded dataset.
  - X_train: (N_tr, 3, 64, 64) array of training images
  - y_train: (N_tr,) array of training labels
  - X_val: (N_val, 3, 64, 64) array of validation images
  - y_val: (N_val,) array of validation labels
  - X_test: (N_test, 3, 64, 64) array of testing images.
  - y_test: (N_test,) array of test labels; if test labels are not available
    (such as in student code) then y_test will be None.
  - mean_image: (3, 64, 64) array giving mean training image
  """
  # First load wnids
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # Use words.txt to get names for each class
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.iteritems():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]

  # Next load training data.
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)
    
    X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        ## grayscale file
        img.shape = (64, 64, 1)
      X_train_block[j] = img.transpose(2, 0, 1)
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # We need to concatenate all training data
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # Next load validation data
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        img.shape = (64, 64, 1)
      X_val[i] = img.transpose(2, 0, 1)

  # Next load test images
  # Students won't have test labels, so we need to iterate over files in the
  # images directory.
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim == 2:
      img.shape = (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)

  y_test = None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)
  
  mean_image = X_train.mean(axis=0)
  if subtract_mean:
    X_train -= mean_image[None]
    X_val -= mean_image[None]
    X_test -= mean_image[None]

  return {
    'class_names': class_names,
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'class_names': class_names,
    'mean_image': mean_image,
  }

# Starting with the NN classifier
class NN(object):

	def __init__(self):
		pass

	def NNClassifierTrain(self,X,y):
		# simply assign the values
		self.X = X
		self.y = y
		self.numExamples = X.shape[0]

	def predict(self,Xtest):
		# predict scores 
		Xtestbig = [[Xtest,]*self.numExamples]
		dist = np.sum(np.abs(self.X - Xtestbig),axis=1)
		prediction = self.y[dist.argmin(0)]
		return prediction

# class definition for the softmax classifier
class SoftMax(object):

	def __init__(self,X,y,Xval,yval):

		# some useful variables
		self.m, self.n = X.shape # m = Nexamples, n = 3072
		self.K = 10 # total classes, by inspection of y

		# append the biases & get X
		self.X = np.append(np.ones((self.m,1)),X,axis=1)

		# append the biases & get Xval
		self.Xval = np.append(np.ones((Xval.shape[0],1)),Xval,axis=1)

		self.y = y
		self.yval = yval

		# convert y into Y matrix of 1s and 0s
		self.Y = np.zeros((self.m,self.K))
		self.Y[np.arange(self.m),self.y] = 1

		# convert yval into Yval matrix of 1s and 0s
		self.Yval = np.zeros((self.yval.shape[0],self.K))
		self.Yval[np.arange(self.yval.shape[0]),self.yval] = 1
		# print(self.Y)

		# initialize the weights which is n * K (for K classes)
		epsilon = 1e-6

		import time
		np.random.seed(int(round(time.time())))

		self.W = epsilon * np.random.random((self.n+1,self.K))
		#self.W = epsilon * np.zeros((self.n+1,self.K))

		# losses list
		self.losses = []
		self.cross_val_losses = []
		
	def BackProp():

		pass

	def CostAndGrad(self,reg):
				
		# define the cost or loss functions		
		H = np.dot(self.X,self.W) # result is m * K (so all examples done) 
		H = np.exp(H) # this is exp(Theta'*x)

		# get the summation of exp() along Ks
		sums = np.sum(H,axis=1)

		# now do the "indicator function" thing...
		temp = H / sums[:,None]
		H = np.multiply(temp,self.Y)

		# don't divide by zero!
		loss = -1/self.m*np.sum(np.log(H[np.nonzero(H)])) + reg/2*np.sum(self.W**2)

		###############################################
		##### This is for cross validation set to plot 
		##### the learning curves
		
		H_CV = np.dot(self.Xval,self.W) # result is m * K (so all examples done) 
		H_CV = np.exp(H_CV) # this is exp(Theta'*x)

		# get the summation of exp() along Ks
		sums_CV = np.sum(H_CV,axis=1)

		# now do the "indicator function" thing...
		temp_CV = H_CV / sums_CV[:,None]
		H_CV = np.multiply(temp_CV,self.Yval)

		# don't divide by zero!
		loss_CV = -1/Xval.shape[0]*np.sum(np.log(H_CV[np.nonzero(H_CV)])) + reg/2*np.sum(self.W**2)

		###############################################

		# now calculate the gradients
		grad = 1/self.m*np.dot(self.X.T,temp-self.Y) + reg*self.W

		return loss, loss_CV, grad 
		
	def train(self,learn_rate=1e-3,Train_iterations=500,reg=1e-2):
		### four steps to go through: ###
		# 1. initialize the weights 
		# 2. Forward prop to calulate loss
		# 3. Back prop to calculate gradients
		# 4. update the weights 

		# initialize the biases as well
		# self.bias = epsilon * np.random.random((self.K,1))

		
		# Training loop 
		for this in range(Train_iterations):
			# compute cost and gradients
			# do an update on the weights and the biases

			cost, loss_cv, grad = self.CostAndGrad(reg)

			# for plotting the learning curves
			self.losses.append(cost)
			self.cross_val_losses.append(loss_cv)

			# learning rate change
			# learn_rate = learn_rate/(Train_iterations)

			# do the Weights update
			self.W -= learn_rate*grad
			#print(cost)

		pypl.plot(self.losses,'r')
		pypl.plot(self.cross_val_losses,'b')
		pypl.ylabel('cost')
		pypl.xlabel('training iteration#')
		pypl.show()
		print('==>Initial cost=',self.losses[0])
		print('==>Final cost=',self.losses[-1])

		print('\nTraining Complete!')

		# get training accuracy
		accuracy = np.mean(np.argmax(np.dot(self.X,self.W),axis=1) == self.y)
		print('==>Accuracy on training set=',accuracy*100,'%')

		# return the model(weights)
		return self.W

	def crossValidate(self,W_optimum):
		accuracy = np.mean(np.argmax(np.dot(self.Xval,W_optimum),axis=1) == self.yval)
		return accuracy

	def predict(self,x):

		pass

def runNearN():
	# nearest neighbour train begins here
	nearNeighbour = NN()
	nearNeighbour.NNClassifierTrain(X,y)
	print('==>NN Training Complete!')

	predictions = []
	print('==>Making predictions...')
	ThisCrossVal = 0
	for i in range(Nexamples):
		prediction = nearNeighbour.predict(Xval[i])
		predictions.append(prediction) 
		print(i+1, 'done')
	print('==>Accuracy on Cross validation test set = ', np.mean(yval == predictions)*100, '%')

def runSoftMax():
	smax = SoftMax(X,y,Xval,yval)
	input('Press "Return" key to begin training...')
	print('Training begun...')
	W_opt = smax.train(Train_iterations=10,learn_rate=1e-2,reg=1e-2)
	accuracy = smax.crossValidate(W_opt)
	print('==>Accuracy on cross validation set=',accuracy*100,'%')

#######################################################

"""Begin by loading the data"""
print('Loading data...')
data = get_CIFAR10_data()
X = data['X_train']
y = data['y_train']
Xval = data['X_val']
yval = data['y_val']
Xtest = data['X_test']
ytest = data['y_test']


"""print(X.shape)
print(y.shape)
print(Xval.shape)
print(yval.shape)
print(Xtest.shape)
print(ytest.shape)"""

# get smaller data sets for prototyping
Nexamples = 4900
Nval = 1000
X = X[:Nexamples,]
y = y[:Nexamples,]
Xval = Xval[:Nval,]
yval = yval[:Nval,]

# reshape some of the data matrices to use
X = np.reshape(X, (Nexamples,3072))
Xval = np.reshape(Xval, (Nval,3072))   # max xtest and xval are 1000

# now do some mean normalization and feature scaling
X = (X - np.mean(X,axis=0)) / np.max(X,axis=0)
Xval = (Xval - np.mean(Xval,axis=0)) / np.max(Xval,axis=0)
#print(np.sum(np.mean(X,axis=0)))

print('data loaded successfully!\n')
#print(y)
"""print(X.shape)
print(y.shape)
print(Xval.shape)
print(yval.shape)
print(Xtest.shape)
print(ytest.shape)"""


####################################################
# Function calls begin here

# Let's see how many examples are we training on
'''print('Statistics of Examples to train on:')
for i in range(10):
	print('Label_%d examples=' %i, np.sum(y == i))'''

runSoftMax()















