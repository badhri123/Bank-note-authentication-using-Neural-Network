Data Set Information:

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.


Attribute Information:

1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous) 
3. curtosis of Wavelet Transformed image (continuous) 
4. entropy of image (continuous) 
5. class (integer) 

banknotesNN.m : This is the program which creates a 2 layer neural network and gets trained using backpropagation.
Performance was observed both on testing and training data to check whether overfitting was occuring.

banknoteclass.txt : This the text file which contains the dataset.The 5th column in each row is the class label of the corresponding example.

sigmoid.m : This function returns the activation value for the hidden and output layer of the neural network.