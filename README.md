# Specch-Based-Emotion-Detection


I here, have used a TESS (Toronto Emotion Speech Set) Dataset from Kaggle. The dataset consists of various emotions' voices. I used this to train my model accordingly.

The Proceeding can be understood as follows : 

# 1. Exploratory Data Analysis

Firstly, I checked for the data. I checked how many number of samples of each voice is there, I used countplot to plot the number count. Also then I used spectogram and waveform diagram to see what do they look like in fourier's relation.

Each of the emotion is significantly different. This makes this dataset ideal. Also, there is no Data Cleaning required.

# 2. Conversion to the MFCC Feature Vector 
Each of the voice file is converted into a MFCC feature vector, which then will be sent to the model for training as we can not send the raw voice for the training purposes.
Librosa is the library with this conversion tool.

# 3.  Model Creation :

The model is a Dense Network. I used keras-tuner to find the best hyperparameters for the model. The model in itself is a good one as it got such a high accuracy. 

The model is then exported in h5 format.

# 4. Application Creation 

I finally deployed the model on a streamlit app.
