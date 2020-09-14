import numpy as np
import subprocess
import pickle
import torch
import os

cifar_10_path = "cifar-10-python.tar.gz"
cifar_100_path = "cifar-100-python.tar.gz"
mnist_path = "mnist.npz"

# URL from: https://www.cs.toronto.edu/~kriz/cifar.html
if not os.path.exists(cifar_10_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", shell=True)
subprocess.call("tar xzfv cifar-10-python.tar.gz", shell=True)

if not os.path.exists(cifar_100_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", shell=True)
subprocess.call("tar xzfv cifar-100-python.tar.gz", shell=True)

# URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
if not os.path.exists(mnist_path):
    subprocess.call("wget https://s3.amazonaws.com/img-datasets/mnist.npz", shell=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Save CIFAR 10
cifar10_train = unpickle('cifar-10-python/train')
cifar10_test = unpickle('cifar-10-python/test')

x_tr = torch.from_numpy(cifar10_train[b'datasets'])
y_tr = torch.LongTensor(cifar10_train[b'fine_labels'])
x_te = torch.from_numpy(cifar10_test[b'datasets'])
y_te = torch.LongTensor(cifar10_test[b'fine_labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar10.pt')

# Save CIFAR 100
cifar100_train = unpickle('cifar-100-python/train')
cifar100_test = unpickle('cifar-100-python/test')

x_tr = torch.from_numpy(cifar100_train[b'datasets'])
y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
x_te = torch.from_numpy(cifar100_test[b'datasets'])
y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar100.pt')

# Save MNIST
f = np.load('mnist.npz')
x_tr = torch.from_numpy(f['x_train'])
y_tr = torch.from_numpy(f['y_train']).long()
x_te = torch.from_numpy(f['x_test'])
y_te = torch.from_numpy(f['y_test']).long()
f.close()

torch.save((x_tr, y_tr), 'mnist_train.pt')
torch.save((x_te, y_te), 'mnist_test.pt')