# kqbc.python 
This is python implementation (and by matlab engine call) for this paper: https://dl.acm.org/citation.cfm?id=2976304 \
NOTE: If git clone doesn't work due to authentication, please download as a ZIP

### Package content:
* test_synth.py - the test that runs the synthetic data experiment 
* kqbc.py - the python functions converted from matlab code
* KQBC.m - matlab original code from the paper


### Matlab engine:
In order to use the KQBC matlab functions do the following:
1) Download Matlab - (for student free version: https://www.mathworks.com/academia/tah-support-program/eligibility.html?s_tid=ac_tahcheck_sv_button)
2) launch Matlab and then in the command window, write:
```
matlabroot 
```
3) This would give you a result with the path to the root folder where the Matlab path is installed. Now based on the operating system being used, open the Command Prompt/ Terminal and move to the location of the python engine setup file like:
```
cd matlabroot/extern/engines/python
```

4) If you list the files in this directory, you should see a file called ‘setup.py’. Now based on the python version installed, you would need to run the following command:
```
python setup.py install
```
In case you have python3.x, replace python with python3. Assuming everything goes correctly, you should see an output like the following on the terminal window.
The test_synth.py import the matlab engine and uses the KQBC matlab functions.

NOTE: In case somthing goes wrong, you can refer to this site for more information: https://medium.com/@pratap_aish/how-do-i-run-my-matlab-functions-in-python-7d2b8b2fefd1


### Running the test:
The synthtetic data test executes the KQBC algorithm for learning a linear classifier in a d-dimensional space. The target classifier is the vector w∗ = (1, 0, . . . , 0) thus the label of an instance x ∈ IRd is the sign of its first coordinate. The instances were normally distributed N (µ = 0, Σ = Id).


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


### Results:
These plots describe the error rate on the test set vs the number of samples the classifier was trained on.
* SVM - random k entries
* KQBC - using the classifier (point in the version space) after T steps (using dot product)

<img src="https://github.com/berryweinst/kqbc.python/blob/master/d_5.png" width="450">
<img src="https://github.com/berryweinst/kqbc.python/blob/master/d_15.png" width="450">
