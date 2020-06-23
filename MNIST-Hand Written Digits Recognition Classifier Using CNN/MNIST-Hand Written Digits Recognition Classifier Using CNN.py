"""
MNIST DATASET - HANDWRITTEN DIGITS RECOGNITION

This is a famous dataset used for hand written digits recognition
We have no need to perform a train test split , since this data set is already pre defined and
pre separated  for our needs..




"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as  plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

single_image = x_train[0]

print(single_image)

print(single_image.shape)

plt.imshow(single_image)
plt.show()

print(y_train)

print(y_train[0])

""" 
As you can see ,  the first x train plot was five , and its corresponding first  y label was also 5 , so 
the x set denotes all the images form 0 to 9 in array , matrix format consisting of pixel values as image
kernels..

On the other hand , y set consist of labels , their actual predictions..

Now we will have to perform one hot label encoding , since the y set needs to be a simple value , instead of known
label value..this will help to convert  a label to a category..

"""

from tensorflow.keras.utils import to_categorical

y_cat_test = to_categorical(y_test)
y_cat_train = to_categorical(y_train)

""" 
Now we have to scale our data to prevent any gradient problem to one scale ,between 0 to 1
"""

print(single_image.max())

print(single_image.min())

x_train = x_train / 255

x_test = x_test / 255

scaled_image = x_train[0]

print(scaled_image)

print(scaled_image.max())

plt.imshow(scaled_image)
plt.show()

""" 
Since the ratio of scaled data in scaled image array is same, it will show the same image as usual

Notice , we can see the data has been scaled now and all matrix or kernel values lie between 0 to 1



Now, we have to reshape our array to let our convolutional neural network know that we are dealing with
single colur channel , grayscale..
"""

# For training set
# batch size = 60000 samples
# height = 28 px
# width =28 px
# colour channel = 1
x_train = x_train.reshape(60000, 28, 28, 1)

# For testing set
# batch size = 60000 samples
# height = 28 px
# width =28 px
# colour channel = 1

x_test = x_test.reshape(10000, 28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4, 4),
                 input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())

# OUTPUT LAYER :

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_Stop = EarlyStopping(monitor='val_loss', patience=1)

model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_Stop])

metrics = pd.DataFrame(model.history.history)

print(metrics.head())

metrics[['loss', 'val_loss']].plot()
plt.show()

metrics[['accuracy', 'val_accuracy']].plot()
plt.show()

print(model.metrics_names)

""" 
The first evaluation will be of loss ,
The second evaluation will be of accuracy  -->
"""

print("The loss and accuracy respectively are : ", model.evaluate(x_test, y_cat_test))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

predictions = model.predict_classes(x_test)
print(predictions)

print(y_test.shape)

print(y_cat_test.shape)

comp_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label ': predictions})
print(comp_df.head(20))

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

""" 
LET US Detect a random image to see our model's performance :

"""

samp_img = x_test[0]
plt.imshow(samp_img.reshape(28, 28))
plt.show()

""" 
We have to reshape our input of new random data :

number of images , height,width,number of colour channels -->

1,28,28,1
"""

pred2 = model.predict_classes(samp_img.reshape(1, 28, 28, 1))
print("The predicted Result of image is : ", pred2)
