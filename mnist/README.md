# Mnist data download

1) Download the MNIST files from Y. LeCun's website, to this mnist folder

* http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
* http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
* http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
* http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

```
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
```

2) Unzip the downloaded gzip archives

```
gunzip t*-ubyte.gz
```
