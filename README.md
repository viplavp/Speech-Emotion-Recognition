# Introduction
The goal of this project is to build **MLP**, and **CNN** model for speech emotion recognition using [Ravdess Emotional Speech](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) audio set.
### Abstract
Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and the associated affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. Emotion recognition is a rapidly growing research domain in recent years. Unlike humans, machines lack abilities to perceive and show emotions. But human-computer interaction can be improved by implementing automated emotion recognition, thereby reducing the need for human intervention. In this project, basic emotions like calm, happy, fearful, disgust etc. are analyzed from emotional speech signals.
<p align="center">
<img src="https://user-images.githubusercontent.com/58522706/130071699-791ac909-c504-44e3-b4b2-ca18184230da.JPG" alt="Speech Emotion Recognition System" width="500" height="350">
</p>

# Dataset
This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.
### File naming convention
Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics
### Filename identifiers
<ul>
<li>Modality (01 = full-AV, 02 = video-only, 03 = audio-only)</li>
<li>Vocal channel (01 = speech, 02 = song)</li>
<li>Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)</li>
<li>Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion</li>
<li>Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")</li>
<li>Repetition (01 = 1st repetition, 02 = 2nd repetition)</li>
 <li>Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)</li>
</ul>

***Filename example: 03-01-06-01-02-01-12.wav***
<ul>
<li>Audio-only (03)</li>
<li>Speech (01)</li>
<li>Fearful (06)</li>
<li>Normal intensity (01)</li>
<li>Statement "dogs" (02)</li>
<li>1st Repetition (01)</li>
<li>12th Actor (12)</li>
</ul>

# Feature Extraction
Feature extraction is very effective in increasing the accuracy and efficiency of machine learning algorithms in emotional speech recognition. The input audio file is reduced to a set of features which represent or summarise the original input. The extracted features are then fed into the neural network to identify the respective emotion. The features which have been used in the project are discussed below.
### Mel Spectogram
An audio signal can be broken down into sine and cosine waves that form the original signal. The frequencies and amplitudes of these representative waves can be used to convert the input signal from the time to the frequency domain. The fast Fourier transform (FFT) is an algorithm that can be used to perform this conversion. The FFT is however performed on a single time window of the input signal. If the frequencies of the representative signals change over time (non-periodic), the change cannot be captured through a single time window conversion. Using the FFT over multiple overlapping windows can be used to construct a spectrogram representing the amplitude of the representative frequencies as they change over time.
<p align="center">
<img src="https://user-images.githubusercontent.com/58522706/130081678-0766da50-535c-4a46-8a6c-757aaedc651f.png" alt="FFT Conversion" width="400" height="450">
</p>

### Mel-frequency cepstral coefficients (MFCCs)
The Fourier spectrum is obtained by using the FFT on an audio signal. Taking the log of the Fourier spectrum’s magnitude and then taking the spectrum of this log through a cosine transformation allows us to calculate the cepstrum of the signal (ceps is the reverse of spec, called so due to being a non-linear ‘spectrum of a spectrum’). The amplitudes of this resulting cepstrum are the cepstral coefficients. Representing the frequencies in terms of the mel scale (discussed above) instead of a linear scale converts the cepstrum to a mel-frequency cepstrum and the cepstral coefficients to mel-frequency cepstral coefficients (MFCCs). The cepstrum represents the rate of change in the different spectrum bands, and the coefficients are widely used in machine learning algorithms for sound processing.
<p align="center">
<img src="https://user-images.githubusercontent.com/58522706/130082550-8378cb95-f9c0-41bb-a3de-2b765214cad5.jpeg" alt="MFCC" width="500" height="350">
</p>

### Chroma 
An octave is a musical interval or the distance between one note and another, which is twice its frequency: A3 (220 Hz) - A4 (440 Hz) – A5 (880 Hz). An octave is divided into 12 equal intervals, and each interval is a different note. The intervals are called chromas or semitones and are powerful representations of audio signals. The calculation of the chromagram is similar to the spectrogram using short-time Fourier transforms, but semitones/pitch classes are used instead of absolute frequencies to represent the signal. The unique representation can show musical properties of the signal which is not picked up by the spectrogram.

<p align="center">
<img src="https://user-images.githubusercontent.com/58522706/130082995-4afec0c8-22a4-4111-9087-5064affa0745.png" alt="Chromagram" width="500" height="350">
</p>

# Model Architecture
<table>
  <tr>
   <th><b>MLP Model</b></td>
 <th><b>CNN Model</b></td>
  </tr>
  <tr>
    <td>A fully connected neural network was deployed to predict the emotion from the features of the sound files.The model consists of 4 hidden layers with a dropout of 0.1 in the first three layers.The first and the second hidden layers consists of 512 neurons while the third layer contained 128 neurons and the fourth hidden layer contained 64 neurons.The model was defined using the keras in-built module Sequential which takes input in the form of layers. 'Relu' activation function was used to calculate the values of parameters from one layer to another layer.A dropout of 0.1 was also added in case of first three hidden layers to avoid overfitting by dropping a node with a 10% probability.In the first two layers 12 regularization was also deployed to overcome the case of overfitting and to effectively tune the parameters.The input layer is of the shape (180,1) and the ouput layer predicts the emotion using the 'softmax' function which gives the probability of each label and the label with the highest probability is selected and given as the answer.</td>
    <td>A CNN was trained and tested on the dataset as well. The model architecture consists of 5 layers in total with first two being 1D convolutional layers and last three being dense fully connected layers.The first two layers contains 32 filters of size 3 each with the 'relu' activation function. Batch normalization was used to speed up the process and to generalize well on unseen data. Two dropout layers were also added in between the convolutional layers to optimize the performance of the model. These two layers were followed by a flatten layer to flatten the ouptut of the convolutional layer followed by 3 fully connected layers with 512, 512, and 256 neurons respectively with 'relu' as activation function. Regularizers and dropout layers were added in between these layers also to avoid the problem of overfitting. Finally, the last dense layer predicted the emotion using 'softmax' activation function.</td>
  </tr>
 </table>


# Accuracy and Confusion Matrix
### MLP Model
Accuracy obtained on training data is  93.24  
Accuracy obtained on test data is  65.83
<table>
  <tr>
    <th>Accuracy Plot</td>
     <th>Confusion Matrix</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/58522706/130087022-03d08046-e00f-4958-a4f9-1ac5b12e9d9c.png" alt="Plot" width="500" height="300"></td>
    <td><img src="https://user-images.githubusercontent.com/58522706/130087290-fbdc14bf-8369-4ee0-b4a9-a261eb14dfba.png" alt="Plot" width="500" height="300"></td>
  </tr>
 </table>
 
 ### CNN Model
 Model
Accuracy obtained on training data is  98.24  
Accuracy obtained on test data is  77.43
<table>
  <tr>
    <th>Accuracy Plot</td>
     <th>Confusion Matrix</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/58522706/130088734-31d9bdc3-56bd-4525-9fb7-79c0e2fb2563.png" alt="Plot" width="500" height="300"></td>
    <td><img src="https://user-images.githubusercontent.com/58522706/130088851-73726194-0a1f-4c79-92bd-daa3daa12bfd.jpeg" alt="Plot" width="500" height="300"></td>
  </tr>
 </table>
 
# References
We referred to the following links of some research papers and articles for our overall work :
<ul>
<li>http://www.jcreview.com/fulltext/197-1594073480.pdf?1625291827</li>
<li>https://www.researchgate.net/publication/341922737_Multimodal_speech_emotion_recognition_and_classification_using_convolutional_neural_network_techniques</li>
<li>https://ijesc.org/upload/bc86f90a8f1d88219646b9072e155be4.Speech%20Emotion%20Recognition%20using%20MLP%20Classifier.pdf</li>
 
