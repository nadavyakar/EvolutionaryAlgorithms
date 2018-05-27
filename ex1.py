import random as rnd
from math import log, exp
import logging
import os
import struct
import numpy as np


def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    for i in xrange(len(lbl)):
        yield get_img(i)

class ActivationSigmoid:
    def __call__(self, IN_VEC):
        return 1. / (1. + np.exp(-IN_VEC))
    def derivative(self, out):
        return (1.0 - out) * out
class ActivationSoftmax:
    def __call__(self, IN_VEC):
        denominator = sum([exp(v) for v in IN_VEC])
        return np.array([exp(v) / denominator for v in IN_VEC ])
    def derivative(self, out):
        raise Error("ERROR: you should not have gotten here ActivationSoftmax")
class ActivationInputIdentity:
    def __call__(self, IN_VEC):
        return IN_VEC
    def derivative(self, out):
        return np.array([.0,])

pic_size=28*28
nclasses=10
train_x = []
train_y = []
i=0
for label,img in read("training"):
    i+=1
    if i>100:
        break
    train_x.append(np.array([float(x) / 255 for x in img.reshape(-1)]))
    train_y.append(label)

test_x = []
i=0
for label,img in read("testing"):
    if i > 10:
        break
    test_x.append(np.array([float(x) / 255 for x in img.reshape(-1)]))

architectures=[[pic_size,50,50,nclasses]]
epocs=[50]
learning_rates=[0.001]
weight_init_boundries=[0.5]


validation_ratio=.1
Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])
logging.basicConfig(filename="nn.log",level=logging.DEBUG)

def init_model(params):
    layer_sizes, weight_init_boundry = params
    return [ np.matrix([[rnd.uniform(-weight_init_boundry,weight_init_boundry) for i in range(layer_sizes[l])] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]
class LossNegLogLikelihood:
    def __call__(self, V, y):
        return -log(np.squeeze(np.asarray(V))[int(y)])
    def derivative_z(self, out, Y):
        return out-Y
loss=LossNegLogLikelihood()
def split_to_valid(train_x,train_y):
    data_set=zip(train_x, train_y)
    train_size=len(data_set)-int(validation_ratio*len(data_set))
    return data_set[:train_size],data_set[:train_size]
sigmoid = ActivationSigmoid()
def fprop(W,X):
    W1=W[0]
    W2=W[1]
    W3=W[2]
    x=X
    z1 = np.dot(W1, x)
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1.T).reshape(-1)
    h2 = sigmoid(z2)
    z3 = np.dot(W3, h2.T).reshape(-1)
    h3 = sigmoid(z3)
    return [h1,h2,h3]
def bprop(W,X,Y,learning_rate):
    out_list=fprop(W,X)
    h3 = out_list[2]
    h2 = out_list[1]
    h1 = out_list[0]
    dz3 = (h3 - np.matrix([Y]))
    W[2] -= learning_rate*np.outer(dz3, h2.T)
    dz2 = np.array(np.dot(W[2].T, (h3 - y).T).reshape(-1).tolist()) * np.array(h2.tolist()) * (1. - np.array(h2.tolist()))
    W[1] -= learning_rate*np.outer(dz2, h1.T)
    dz1 = np.array(np.dot(W[1].T, dz2.T).reshape(-1).tolist()) * np.array(h1.tolist()) * (1. - np.array(h1.tolist()))
    W[0] -= learning_rate*np.outer(dz1, X.T)
def validate(W,valid):
    sum_loss= 0.0
    correct=0.0
    for X, y in valid:
        out = fprop(W,X)
        sum_loss += loss(out[-1],y)
        if out[-1].argmax() == y:
            correct += 1
    return sum_loss/ len(valid), correct/ len(valid)
def train(W,train_x,train_y,learning_rate,starting_epoc,ending_epoc,avg_loss_list,avg_acc_list):
    train,valid=split_to_valid(train_x,train_y)
    for e in range(starting_epoc,ending_epoc):
        logging.debug("starting epoc {}".format(e))
        rnd.shuffle(train)
        for X,y in train:
            bprop(W,X,Y[y],learning_rate)
        avg_loss,acc=validate(W, valid)
        print("epoc {} avg_loss {} acc {}".format(e,avg_loss,acc))
        avg_loss_list.append(avg_loss)
        avg_acc_list.append(acc)
    return avg_loss_list,avg_acc_list
def test(W,test_x):
    for X in test_x:
        p=np.squeeze(np.asarray(fprop(W, X)[-1]))
        # print("p {} y_hat {}".format(p,p.argmax()))

for weight_init_boundry in weight_init_boundries:
     for architecture in architectures:
         for learning_rate in learning_rates:
             W=init_model((architecture, weight_init_boundry))
             avg_loss_list = []
             avg_acc_list = []
             starting_epoc=0
             for ending_epoc in epocs:
                 avg_loss_list, avg_acc_list = train(W,train_x,train_y,learning_rate,starting_epoc,ending_epoc,avg_loss_list,avg_acc_list)
                 starting_epoc = ending_epoc
                 test(W,test_x)