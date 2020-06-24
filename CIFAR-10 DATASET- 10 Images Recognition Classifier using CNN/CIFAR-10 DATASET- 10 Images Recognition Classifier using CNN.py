"""
CIFAR-10  DATASET CONVOLUTION NEURAL NETWORK - CNN


"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_train.shape)

""" 
 X_TRAIN DATA :
 
(50000, 32, 32, 3) - >
50000- samples
32- height 
32 - width 
3-colour channels - RGB
"""

for x in range(0,5):
    plt.imshow(x_train[x])
    plt.show()




""" 
Understanding the original labels:

The label data is just a list of 10,000 numbers ranging from 0 to 9, which corresponds to each of the 10 classes in CIFAR-10.

airplane : 0
automobile : 1
bird : 2
cat : 3
deer : 4
dog : 5
frog : 6
horse : 7
ship : 8
truck : 9
"""

def get_label_from_test(y_test):
    for i in (y_test):
        if i == 0:
            valt='Airplane'
        elif i == 1:
            valt='Automobile'
        elif i == 2:
            valt='Bird'
        elif i == 3:
            valt='Cat'
        elif i == 4:
            valt='Deer'
        elif i == 5:
            valt='Dog'
        elif i == 6:
            valt='Frog'
        elif i == 7:
            valt='House'
        elif i == 8:
            valt='Ship'
        else:
            valt='Truck'
    return str(valt)


def get_label_from_train(y_train):
    for i in (y_train):

        if i == 0:
            valt = 'Airplane'
        elif i == 1:
            valt = 'Automobile'
        elif i == 2:
            valt = 'Bird'
        elif i == 3:
            valt = 'Cat'
        elif i == 4:
            valt = 'Deer'
        elif i == 5:
            valt = 'Dog'
        elif i == 6:
            valt = 'Frog'
        elif i == 7:
            valt = 'House'
        elif i == 8:
            valt = 'Ship'
        else:
            valt = 'Truck'
    return str(valt)
""" 
PRE-PROCESSING DATA :

Scaling Of Data :
"""

print(x_train.max())

x_train=x_train/255

x_test=x_test/255

print(y_test)

""" 
These are not continuous labels, these are categorical labels , based on photos

Lets us convert them to binary set of integers using to categorical for multi class classification :

"""

from tensorflow.keras.utils import to_categorical

y_cat_train=to_categorical(y_train,10)

y_cat_test=to_categorical(y_test,10)



""" 
Building A CNN Model :


In CONV . LAYER -> Input Shape : input_shape=(32,32,3) from x_train shape

Also , since 32*32*3 is 3072 , hence , we have  larger number of parameters, there fore , i 
am gonna create 2 layers of convolutional , followed by pooling layers..

since our data is complex , i am adding a layer of dense layer of 256 neurons as well..before final output
node layer..

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten

model=Sequential()


#CONVOLUTIONAL LAYER :
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))

#POOLING LAYERS :
model.add(MaxPool2D(pool_size=(2,2)))


#CONVOLUTIONAL LAYER :
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))

#POOLING LAYERS :
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(256,activation='relu'))

#OUTPUT LAYER :
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

from tensorflow.keras.callbacks import EarlyStopping
early_Stop=EarlyStopping(monitor='val_loss',patience=2)

model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_Stop])



metrics=pd.DataFrame(model.history.history)
print(metrics)

print(metrics.columns)

metrics[['loss','val_loss']].plot()
plt.title('Loss VS Val_Loss')
plt.show()

metrics[['accuracy','val_accuracy']].plot()
plt.title('Accuracy VS Val_Accuracy')
plt.show()

eval=model.evaluate(x_test,y_cat_test)
print("Loss and accuracy of model respectively are : ",eval)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

predictions=model.predict_classes(x_test)

print(classification_report(y_test,predictions))

plt.figure(figsize=(12,6))
sns.heatmap(confusion_matrix(y_test,predictions))
plt.show()

# Comparison Dataframe :

comp_df = pd.DataFrame({'Actual Label': y_test.reshape(10000, ), 'Predicted Label': predictions})
print(comp_df.head(10))
l1 = []

for i in predictions:
    if i == 0:
        l1.append('Airplane')

    elif i == 1:
        l1.append('Automobile')

    elif i == 2:
        l1.append('Bird')

    elif i == 3:
        l1.append('Cat')

    elif i == 4:
        l1.append('Deer')

    elif i == 5:
        l1.append('Dog')

    elif i == 6:
        l1.append('Frog')

    elif i == 7:
        l1.append('House')

    elif i == 8:
        l1.append('Ship')

    else:
        l1.append('Truck')

comp_df['Predicted Label Name'] = l1
print(comp_df.head(20))



""" 
Prediction ON Random Data :
"""

sampl_img=x_test[16]

plt.imshow(sampl_img)
plt.show()

print('True Label Is :', y_test[16])


pred2=model.predict_classes(sampl_img.reshape(1,32,32,3))

print('Predicted Label IS : ',pred2)

print("The predicted label is : ",get_label_from_test(pred2))
print("The actual label is : ", get_label_from_test(y_test[16]))