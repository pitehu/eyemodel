# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:23:17 2018

@author: txh160830
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:14:27 2018

@author: txh160830
"""
#import configg
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#import tensorflow as tf
import glob
import json, re
import numpy as np
import random
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, Activation,MaxPooling2D,Dropout, Flatten, Dense, BatchNormalization,SpatialDropout2D
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#import tensorflow as tf
import cv2
from keras.callbacks import Callback

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'Weights\Weight20180928\weights_monewdel2%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1
#from keras import applications
#import tensorflow as tf

#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


#img_width, img_height = 120, 80
#train_data_dir = 'train\'
#validation_data_dir = 'test\'
steps_per_epoch=47544
batch_size=32
epochs=8

bigcount1=0
addree=glob.glob(r'C:\Users\txh160830\eye\UnityEyes_Windows\imgs1\*.*')
def eva_mol(numofsample):
    addree=glob.glob(r'C:\Users\txh160830\eye\UnityEyes_Windows\imgs1\*.*')

    x=[]
    y=[]
    x_raw=[]
    for i in range(0,numofsample*2+1,2):
    #for i in range(0,10,2):
        #print(a[i])
            #x1=img_to_array(load_img(addree[i]))
            #print(i)
            x1=cv2.imread(addree[i])
            #plt.imshow(x1)
            #print(type(x1))
            x2=x1[150:300, 240:464]
            x_raw.append(x1)    
            x.append(x2)
            file1=open(addree[i+1], "r" )
            y1=json.load(file1)['iris_2d']
            file1.close()
            y2=np.zeros((32,2))
            #print(y)
            count=0
            for temp in y1:
                temp1=re.findall("\d+\.\d+", temp)
                temp2=[float(i) for i in temp1]
                del temp2[2]
                #print(temp2)
                y2[count,0]=temp2[0]
                y2[count,1]=temp2[1]
                count+=1
                #print('count=',count)
                #print(y2)
            #y2.reshape((64,1))
            y.append(y2)
            #print('i=',i)
                    # create numpy arrays of input data
                    # and labels, from each line in the fil
                #print(x)
            #print(y)
    #            print(x)
    #                x1, x2, y = process_line(line)
    x=np.asarray(x)
    x_raw=np.asarray(x_raw)
    y=np.asarray(y)
    y=y.reshape((-1,64))
    x=x.reshape((-1,150,224,3))
    #numofsample=(ub-lb)//2
    return (x,x_raw,y)

def test_data_prepo(lb,ub):
    x=[]
    y=[]
    x_raw=[]
    x2=[]
    #y_prepo=[]
    for i in range(lb,ub,2):
    #for i in range(0,10,2):
        #print(a[i])
            #x1=img_to_array(load_img(addree[i]))
            file1=open(addree[i+1], "r" )
            y1=json.load(file1)['iris_2d']
            file1.close()
            y2=np.zeros((32,2))
            #print(y)
            count=0
            for temp in y1:
                temp1=re.findall("\d+\.\d+", temp)
                temp2=[float(i) for i in temp1]
                del temp2[2]
                #print(temp2)
                y2[count,0]=temp2[0]
                y2[count,1]=480-temp2[1]
                count+=1
                #print('count=',count)
                #print(y2)
            #y2.reshape((64,1))
            #y2=480-y2[:,0]
                           
            y.append(y2)
            
    y1=np.asarray(y)
    count1=0
    for i in range(lb,ub,2):        
            #y_prepo=np.append(y_prepo,y2)
            x1=(cv2.imread(addree[i]))
            x_raw.append(x1)
            x1=cv2.resize(x1,(320,240))
            
            #x3=np.zeros(160,120)
            #x2=tf.random_crop(x1,(160,120,3))
            #x2=x1[150:300, 240:464]
            #rando=random.uniform(0.25, 0.75)
            rando=random.uniform(0.20, 0.90)
            newwidth=round((rando*320 / 2.) * 2)
            newheight=round((rando*240 / 2.) * 2)
            x3=cv2.resize(x1,(newwidth,newheight))
            #y1[count1,:,0]=y1[count1,:,0]*round((rando*320 / 2.) * 2)/320*0.5
            #y1[count1,:,1]=y1[count1,:,1]*round((rando*240 / 2.) * 2)/240*0.5
#            y1[count1,:,0]=y1[count1,:,0]*round((rando*320 / 2.) * 2)/320*0.5
#            y1[count1,:,1]=y1[count1,:,1]*round((rando*240 / 2.) * 2)/240*0.5
            y3=np.asarray(y1)
            x2.append(x3)
            if newwidth<=160:
                y1[count1,:,0]=y1[count1,:,0]*newwidth/320*0.5
                y1[count1,:,1]=y1[count1,:,1]*newheight/240*0.5
                padhl=(120-newheight)//2
                padhr=120-padhl-newheight
                padwl=(160-newwidth)//2
                padwr=160-padwl-newwidth
                x3=np.pad(x3,((padhl,padhr),(padwl,padwr),(0,0)),'constant',constant_values=255)
                #y_prepo=np.asarray(y_prepo)
                #print(y_prepo.shape)
                y1[count1,:,0]=y1[count1,:,0]+padwl
                y1[count1,:,1]=y1[count1,:,1]+padhl
                #print('padheight',count1,padhl)
                #print('padweidth',count1,padwl)
                count1+=1
                print('I am small')
            else:
                #print('newheight',count1,newheight)
                #print('newweight',count1,newwidth)
                y1[count1,:,0]=y1[count1,:,0]*newwidth/320*0.5
                y1[count1,:,1]=y1[count1,:,1]*newheight/240*0.5
                crophl=int(newheight/2-60)
                crophr=int(newheight/2+60)
                cropwl=int(newwidth/2-80)
                cropwr=int(newwidth/2+80)
                x3=x3[crophl:crophr,cropwl:cropwr]
                y1[count1,:,0]=y1[count1,:,0]-cropwl
                y1[count1,:,1]=y1[count1,:,1]-crophl     
                count1+=1
            x.append(x3)
            
            #print('i=',i)
                    # create numpy arrays of input data
                    # and labels, from each line in the fil
                #print(x)
            #print(y)
    #            print(x)
    #                x1, x2, y = process_line(line)
    x=np.asarray(x)
    x_raw=np.asarray(x_raw)
    y=np.asarray(y)
    y1=np.asarray(y1)
    #y=y.reshape((-1,64))
    #x=x.reshape((-1,150,224,3))
    numofsample=(ub-lb)//2
    
    return (x,x_raw,x2,y,y1,y3,numofsample)
addr=glob.glob(r'E:\Data\MSP-Gaze\eye_image\*\*.jpg')

    
def test_on_gaze_img(lb,up,addr):
    xl=[]
    xr=[]
    #addr=glob.glob(r'E:\Data\MSP-Gaze\eye_image\*\*.jpg')
    
    for i in range(lb,up):
        x1=cv2.imread(addr[i])
        x2=x1[:,0:50]
        x3=x1[:,50:100]
        #x2=cv2.resize(x2,(160,120))
        #x3=cv2.resize(x3,(160,120))
        x2=np.pad(x2,((47,48),(55,55),(0,0)),'constant',constant_values=255)
        x3=np.pad(x3,((48,47),(55,55),(0,0)),'constant',constant_values=255)
        xl.append(x2)
        #print(xl)
        xr.append(x3)
    xl=np.asarray(xl)
    xr=np.asarray(xr)
    return xl,xr
        
    

def test_data(lb,ub):
    x=[]
    y=[]
    x_raw=[]
    for i in range(lb,ub,2):
    #for i in range(0,10,2):
        #print(a[i])
            #x1=img_to_array(load_img(addree[i]))
            x1=(cv2.imread(addree[i]))
            x1=cv2.resize(x1,(320,240))
            
            x2=x1[150:300, 240:464]
            x_raw.append(x1)    
            x.append(x2)
            file1=open(addree[i+1], "r" )
            y1=json.load(file1)['iris_2d']
            file1.close()
            y2=np.zeros((32,2))
            #print(y)
            count=0
            for temp in y1:
                temp1=re.findall("\d+\.\d+", temp)
                temp2=[float(i) for i in temp1]
                del temp2[2]
                #print(temp2)
                y2[count,0]=temp2[0]
                y2[count,1]=temp2[1]
                count+=1
                #print('count=',count)
                #print(y2)
            #y2.reshape((64,1))
            y.append(y2)
            #print('i=',i)
                    # create numpy arrays of input data
                    # and labels, from each line in the fil
                #print(x)
            #print(y)
    #            print(x)
    #                x1, x2, y = process_line(line)
    x=np.asarray(x)
    x_raw=np.asarray(x_raw)
    y=np.asarray(y)
    y=y.reshape((-1,64))
    x=x.reshape((-1,150,224,3))
    numofsample=(ub-lb)//2
    return (x,x_raw,y,numofsample)

def testplot(lb,ub):
    x,x_raw,y,numofsample=test_data(lb,ub)
    #fig=plt.figure(figsize=(20, 15))
    column=2
    row=numofsample//2
    y_model=model.predict(x)
    y_model=y_model.reshape(-1,32,2)
    y=y.reshape(-1,32,2)
    print(y_model.shape)
    for i in range(0,numofsample):
        fig, ax = plt.subplots()
        poly=Polygon(y_model[i])
        plt.imshow(x_raw[i-1])
        ax.add_patch(poly)
        plt.show()
        print('baseline')
        fig1, ax1 = plt.subplots()
        poly1=Polygon(y[i])
        plt.imshow(x_raw[i-1])
        ax1.add_patch(poly1)
        plt.show()
    
def testplot1(x_raw,y_predict,y_real,num):
    for i in range(0,num):
#        fig, ax = plt.subplots()
#        plt.imshow(x_raw[i])
#        ax.fill(y_predict[i,:,:1],y_predict[i,:,:2])
#        plt.show()
#        print('baseline')
#        fig1, ax1 = plt.subplots()
#        plt.imshow(x_raw[i])
#        ax1.fill(y_real[i,:,:1],y_real[i,:,:2])
#        plt.show()
        
        fig, ax = plt.subplots()
        plt.imshow(x_raw[i])
        poly=Polygon(y_real[i])
        ax.add_patch(poly)
        #ax.fill(y_predict[i,:,:1],y_predict[i,:,:2])
        plt.show()
#        print('baseline')
#        fig1, ax1 = plt.subplots()
#        plt.imshow(x_raw[i])
#        ax1.fill(y_real[i,:,:1],y_real[i,:,:2])
#        plt.show()
        
        


def generate_arrays_from_file():
    
    while True:
        global bigcount1
        global addree
        #print(bigcount1)
        #print(a)
        x=[]
        y=[]
        y3=[]
        #y3=[]
    #    for i in range(0,4,2):
    #        print(i)
        for i in range(bigcount1,bigcount1+batch_size*2,2):
    #for i in range(0,10,2):
        #print(a[i])
            x=[]
            y=[]
    #for i in range(0,10,2):
        #print(a[i])
            #x1=img_to_array(load_img(addree[i]))
            file1=open(addree[i+1], "r" )
            y1=json.load(file1)['iris_2d']
            file1.close()
            y2=np.zeros((32,2))
            #print(y)
            count=0
            for temp in y1:
                temp1=re.findall("\d+\.\d+", temp)
                temp2=[float(i) for i in temp1]
                del temp2[2]
                #print(temp2)
                y2[count,0]=temp2[0]
                y2[count,1]=480-temp2[1]
                count+=1
                #print('count=',count)
                #print(y2)
            #y2.reshape((64,1))
            #y2=480-y2[:,0]
                           
            y.append(y2)
            #print(i)
            #print(y)
            y3.append(y)
            #print('i in first loop',i)
        #print(y1)
        count1=0
        #print(y1)
        y3=np.asarray(y3)
        
        #print()
        #print(y3.shape)
        for i in range(bigcount1,bigcount1+batch_size*2,2):      
            #y_prepo=np.append(y_prepo,y2)
            #print('i in second loop',i)
            x1=(cv2.imread(addree[i]))
            #x.append(x1)
            x1=cv2.resize(x1,(320,240))
            #print('count1',count1)
            #x3=np.zeros(160,120)
            #x2=tf.random_crop(x1,(160,120,3))
            #x2=x1[150:300, 240:464]
            #rando=random.uniform(0.25, 0.75)
            rando=random.uniform(0.25, 2)
            newwidth=round((rando*320 / 2.) * 2)
            newheight=round((rando*240 / 2.) * 2)
            x3=cv2.resize(x1,(newwidth,newheight))
            #y1[count1,:,0]=y1[count1,:,0]*round((rando*320 / 2.) * 2)/320*0.5
            #y1[count1,:,1]=y1[count1,:,1]*round((rando*240 / 2.) * 2)/240*0.5
#            y1[count1,:,0]=y1[count1,:,0]*round((rando*320 / 2.) * 2)/320*0.5
#            y1[count1,:,1]=y1[count1,:,1]*round((rando*240 / 2.) * 2)/240*0.5
            #y3=np.asarray(y1)
            #x2.append(x3)
            #print("I gpt here",i)
            if newwidth<=160:
                y3[count1,:,:,0]=y3[count1,:,:,0]*newwidth/320*0.5
                y3[count1,:,:,1]=y3[count1,:,:,1]*newheight/240*0.5
                padhl=(120-newheight)//2
                padhr=120-padhl-newheight
                padwl=(160-newwidth)//2
                padwr=160-padwl-newwidth
                x3=np.pad(x3,((padhl,padhr),(padwl,padwr),(0,0)),'constant',constant_values=255)
                #y_prepo=np.asarray(y_prepo)
                #print(y_prepo.shape)
                y3[count1,:,:,0]=y3[count1,:,:,0]+padwl
                y3[count1,:,:,1]=y3[count1,:,:,1]+padhl
                #print('padheight',count1,padhl)
                #print('padweidth',count1,padwl)
                count1+=1
            else:
                #print('newheight',count1,newheight)
                #print('newweight',count1,newwidth)
                y3[count1,:,:,0]=y3[count1,:,:,0]*newwidth/320*0.5
                y3[count1,:,:,1]=y3[count1,:,:,1]*newheight/240*0.5
                crophl=int(newheight/2-60)
                crophr=int(newheight/2+60)
                cropwl=int(newwidth/2-80)
                cropwr=int(newwidth/2+80)
                x3=x3[crophl:crophr,cropwl:cropwr]
                y3[count1,:,:,0]=y3[count1,:,:,0]-cropwl
                y3[count1,:,:,1]=y3[count1,:,:,1]-crophl     
                count1+=1
#            #print("success")
            x.append(x3)
#            print('i in second loop',i)
#
#            #print(i)
#            #print('i=',i)
#                    # create numpy arrays of input data
#                    # and labels, from each line in the fil
#                #print(x)
#            #print(y)
#    #            print(x)
#    #                x1, x2, y = process_line(line)
#        #print("success")
        x=np.asarray(x)
#    #x_raw=np.asarray(x_raw)
#    #y=np.asarray(y)
#    #y1=np.asarray(y1)
        y3=y3.reshape((-1,64))
    
    #print()
        
        #print(x.shape)
        #print(y3.shape)
        yield (x,y3)
        bigcount1=bigcount1+batch_size*2
    #y=y.reshape((-1,64))
    #x=x.reshape((-1,150,224,3))
    #numofsample=(ub-lb)//2
    
model=Sequential()
model.add(Conv2D(64,(3,3),padding='same',input_shape=(120,160,3),name='main_input'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(64, (3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(128, (3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(512, (3, 3),padding='same'))
#model.add(SpatialDropout2D(0.5))
#model.add(Conv2D(512, (3, 3),padding='same'))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(512, (3, 3),padding='same'))
#model.add(SpatialDropout2D(0.5))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5)) 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))


model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['accuracy'])
model.summary()

his=model.fit_generator(generate_arrays_from_file(),callbacks=[WeightsSaver(model, 1000)], steps_per_epoch=steps_per_epoch, epochs=epochs,max_queue_size=2)
    
    #scores = model.evaluate(x_test, y_test, verbose=1)
    #print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
    
    
        