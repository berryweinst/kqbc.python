# kqbc.python 
This is python implementation (and by matlab engine call) for this paper: https://dl.acm.org/citation.cfm?id=2976304

### Package content:
application:
* test_synth.py - the test that runs the synthetic data experiment 
* K



### Remarks:
* This app runs Mxnet graph but you have to install pytorch package to handle the image processing and bbox handeling.
* This app uses batch size of 8 and input size of 960x960.
* This app includes preprocessing and post processing utilities, including the data handler. 


### Run the app on the images smaples provided:
```
python yolo_app.py --images ./imgs --write_images_with_predictions 1 --det det --reso 960 --bs 8
```

### Dependencies:
Python 3.5
OpenCV
PyTorch 0.4
