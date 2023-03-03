#Import required libraries 
import keras #library for neural network
import pandas as pd #loading data in table form  
import matplotlib.pyplot as plt #visualisation
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import Preprocessing 

dataset_col = ['wife_age', 'wife_education', 'husband_education', 'number_of_children', 'wife_religion','wife_working', 'husband_occupation', 'standard_of_living_index', 'media_exposure', 'contraceptive_method']

df = pd.read_csv('data\cmc.data',names=dataset_col)     #read data

#Preprocessing data
df = Preprocessing.preprocess(df)    

df=df.iloc[np.random.permutation(len(df))]

X=df.iloc[:,:9].values      #create array X contain features of all record
y=df.iloc[:,9].values       #create array y contain class attribute of all record 

print()
# print("Unique values after data preprocessing")
# for i in df.columns:
#     # print(i)
#     if(i != 'contraceptive_method'):
#         print('Attribute:', i,'\t' ,df[i].unique())
#     else:
#         print('Attribute:', i,'\t' ,np.unique(y,axis=0))


# print("Shape of X",X.shape)
# print("Shape of y",y.shape)
# print("Examples of X\n",X[:3])
# print("Examples of y\n",y[:3])


#using 10-fold from sklearn
kfold = KFold(n_splits=10, shuffle=True, random_state=23)
accuracy_scores = []    #for collecting accuracy 
loss_scores = []        #for collecting loss

for train, test in kfold.split(X, y):       #loop จำนวน train and test that got from kfold 
    # print(train,test)
    X_train = X[train]      
    X_test = X[test] 
    y_train = y[train]
    y_test = y[test]

    
    y_train = np_utils.to_categorical(y[train], num_classes=3)
    y_test =np_utils.to_categorical(y[test], num_classes=3)

    # print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
    # print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])

    #select optimizer using Adam alforithm
    opt = Adam(learning_rate=0.005)
    model=Sequential()
    model.add(Dense(17,input_shape=(9,),activation='sigmoid'))
    # model.add(Dropout(0.2))
    model.add(Dense(33,activation='sigmoid'))
    # model.add(Dropout(0.2))
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dense(3,activation='softmax'))            #Output Node, softmax because last layer node are one-hot vector
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])     #create model

    # model.summary()       #show model structure
    
    #collecting record that got from train model
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=30,epochs=200,verbose=1) 
    """
    model.fit ->
    ใช้ X_train เป็น features และ y_train เป็น target
    วัดผลโมเดลโดยใช้ validation_data ซึ่งก็คือ X_test (features), y_test(target)
    batch ป้อนข้อมูลให้โมเดล(ยิ่งเยอะยิ่งเทรนเร็ว) 
    epoch ปรับทั้งหมด 150 ครั้ง
    verbose = 0 ไม่แสดงผลการรันโมเดลบนหน้าจอ (verbose=1 คือแสดง)
    """
        

    # prediction=model.predict(X_test)       #train test model and collect prediction
    # print(prediction)
    # length=len(prediction)
    # y_label=np.argmax(y_test,axis=1)        #calculate true value from one-hot vector
    # print(y_label)
    # predict_label=np.argmax(prediction,axis=1)  #calculate prediction
    # print(predict_label)
    # print(y_label==predict_label)
    # accuracy=np.sum(y_label==predict_label)/length * 100    #calculate accuracy(sum only prediction is correct)

    y_pred = model.predict(X_test)
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
    print()

    accuracy = accuracy_score(pred,test)
    print('Accuracy is:', accuracy*100)
    print('Loss:', history.history['val_loss'][-1])
    print("Accuracy of the fold",accuracy )
    loss_scores.append(history.history['val_loss'][-1])     #store loss in array
    accuracy_scores.append(accuracy)                        #store accuracy in array
    # k+=1

for i in range(1,11):       #loop printing loss and accuracy fold by fold
    print('Fold:', i)
    print("Loss:",loss_scores[i-1] ,end='')
    print(" Accuracy:",accuracy_scores[i-1] )
    print('--------------------------------------------------------')
    print()

print("Mean loss over 10 folds:", np.mean(loss_scores))
print("Mean accuracy over 10 folds:", np.mean(accuracy_scores))
print()
