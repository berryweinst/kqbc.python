# kqbc.python 
This is python implementation for this paper: https://dl.acm.org/citation.cfm?id=2976304 \

### Package content:
* test_synth.py - the test that runs the synthetic data experiment 
* kqbc.py - the python functions converted from matlab code
* matlab/KQBC.m - matlab original code from the paper
* matlab/hit_n_run.m - matlab original code from the paper


### Data
#### Synthetic data
The synthtetic data test executes the KQBC algorithm for learning a linear classifier in a d-dimensional space. The target classifier is the vector w∗ = (1, 0, . . . , 0) thus the label of an instance x ∈ IRd is the sign of its first coordinate. The instances were normally distributed N (µ = 0, Σ = Id).
#### Mnist data
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. 
For out experiment I degenerated the dataset into 2 classes (1,-1), which are one of the figures and all the rest, and balanced between them by eliminating most of the other classes samples. The final data was a sub version of the original one, consinting of 1000 training examples for each class, so 2000 in total, and 200 for each test class in the same manner.


### Running the test:
#### Flags:
* steps - defines the number of repetitions of the experiment for statistical significance. 
* dim - the dimention of the samplse and classifiers
* samples - a list of number of samples to train on
* hnr_ - number of hit and run walks inside the polytope

Command line example:
```
python test_synth.py --steps 20 --dim 5 --hnr_iter 21 --plot
```

### Dependencies:
Python \
matlab \
numpy \
matplotlib


## Results:
These plots describe the error rate on the test set vs the number of samples the classifier was trained on.
* SVM - random k entries
* KQBC - using the classifier (point in the version space) after T steps (using dot product)

### Synthetic data (5 dimensions - logarithmic and linear scale):
<img src="https://github.com/berryweinst/kqbc.python/blob/master/plots/d_5_log.png" width="550">
<img src="https://github.com/berryweinst/kqbc.python/blob/master/plots/d_5_linear.png" width="550">

### Synthetic data (15 dimensions - logarithmic and linear scale):
<img src="https://github.com/berryweinst/kqbc.python/blob/master/plots/d_15_log.png" width="550">
<img src="https://github.com/berryweinst/kqbc.python/blob/master/plots/d_15_linear.png" width="550">

### Mnist data (only 2 catagories - 1 class vs all):
#### Log scale
<img src="https://github.com/berryweinst/kqbc.python/blob/master/plots/rbf_log.png" width="550">

#### Linear scale
<img src="https://github.com/berryweinst/kqbc.python/blob/master/plots/rbf_linear.png" width="550">

It is clear that we get exponential improvenet when the classifier is extracted from KQBC algorithm.

