import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add, AveragePooling2D
from tensorflow.keras.models import Model
#cifar10
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import math
#load dataset
(trainX,trainY),(testX,testY)=cifar10.load_data()
trainX=trainX.astype('float32')
testX=testX.astype('float32')
trainX=trainX/255.0
testX=testX/255.0
trainY=to_categorical(trainY)
testY=to_categorical(testY)
    

def Wide_residual_network(input_shape,depth,width,dropout_rate=0.3,classes=10):
    inputs=tf.keras.Input(shape=input_shape)
    k=width
    n=(depth-4)//6
    filters=[16*k,32*k,64*k]
    x=Conv2D(16,kernel_size=(3,3),padding='same')(inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    for i in range(3):
        for j in range(n):
            stride = 2 if (i == 1 and j==0) or (i == 2 and j == 0) else 1
            x=wide_dropout_block(x,filters[i],dropout=0.3,stride=stride)
    x=BatchNormalization()(x)
    x=AveragePooling2D(pool_size=(8,8))(x)
    x=Flatten()(x)
    x=Dense(classes,activation='softmax')(x)
    
    model=Model(inputs,x)
    return model
    

    


def wide_dropout_block(inputs,filters,dropout=0.3,stride=1):
    x=BatchNormalization()(inputs)
    x=Activation('relu')(x)
    x=Conv2D(filters,kernel_size=(3,3),padding='same',strides=(stride,stride))(inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.3)(x)
    x=Conv2D(filters,kernel_size=(3,3),padding='same')(x)
    
    if stride!=1 or inputs.shape[-1]!=filters:
        inputs=Conv2D(filters,kernel_size=(1,1),strides=(stride,stride),padding='valid')(inputs)
        inputs=BatchNormalization()(inputs)

    #add
    x=Add()([x,inputs])
    


    return x

#learning rate schedule
def step_decay(epoch):
    initial_lrate=0.1
    drop=0.5
    epochs_drop=20.0
    lrate=initial_lrate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
#build model
model=Wide_residual_network(input_shape=(32,32,3),depth=28,width=10,dropout_rate=0.3)
model.summary()
#compile model
opt=SGD(lr=0.1,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
#learning rate schedule
lrate=LearningRateScheduler(step_decay)
callbacks_list=[lrate]
#train model
history=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=200,batch_size=128,callbacks=callbacks_list,verbose=1)
#evaluate model
_,acc=model.evaluate(testX,testY,verbose=1)
print('Accuracy:%.3f'%(acc*100.0))



