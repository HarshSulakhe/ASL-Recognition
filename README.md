# ASL-Prediction
A deep learning approach to classify alphabets of the American Sign Language using transfer learning

# Dependencies
- fastai 1.0.61
- Python 3.7.6

# Dataset

[Sign Language MNIST on Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist/)<br>
The data was in the form of a csv file with each column containing the pixel value of each image.
I first converted these pixel values and stored them locally as images so it would be easier to create a DataBunch using the fastai library. The script that I used to do the same is **images.py**

# Approach
- Used transfer learning with and without finetuning for both resnet architectures (**ResNet50** and **ResNet34**).<br>
- Data augmentation was done using fastai's ```get_transforms()``` method with no vertical flipping and a reduction to the default ```max_rotate``` parameter.<br>
- The models were first trained without finetuning on 8 cycles with ```max_lr = slice(1e-3)```. The model layers were then unfrozen and trained on 5 further cycles with the same learning rate.

# Results

### Resnet50
- Without finetuning:<br>
Validation Accuracy = **87.41%**<br>
top-5 Accuracy = **99.62%**
- With finetuning:<br>
Validation Accuracy = **99.27%**<br>
top-5 Accuracy = **100%**

### Resnet34
- Without finetuning:<br>
Validation Accuracy = **83.79%**<br>
top-5 Accuracy = **99.12**%
- With finetuning:<br>
Validation Accuracy = **99.90%**<br>
top-5 Accuracy = **100%**

# To-Do
- [ ] Try integrating an OpenCV window to perform real time tracking
