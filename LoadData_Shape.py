# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from sklearn.cluster import MiniBatchKMeans
import cv2
from scipy.cluster.vq import kmeans,vq
from scipy.spatial import distance
import random
from sklearn.preprocessing import MinMaxScaler,minmax_scale
import pickle
import imutils

!pip install opencv-contrib-python==4.4.0.44

"""## Dataset Preparation"""

from google.colab import drive
drive.mount('/content/drive')

import glob
mypath = "/content/drive/My Drive/DIP/DIP_flowers"
# l = glob.glob("/content/drive/My Drive/DIP/DIP_flowers/image_*")
# random.shuffle(l)
# random.shuffle(l)
# random.shuffle(l)
# ll = [pd.read_csv(path) for path in l]
from os import walk
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
for i in range(len(f)):
  f[i] = mypath + "/" + f[i]
df = pd.DataFrame(f,columns=['img'])
# f = open("/content/drive/My Drive/DIP/DIP_flowers/txt/files.txt")
# data = f.read().splitlines()

df['label']=''
df['label'].iloc[0:80]="Daffodil"
df['label'].iloc[80:160]="Snowdrop"
df['label'].iloc[160:240]="LillyValley"
df['label'].iloc[240:320]="Bluebell"
df['label'].iloc[320:400]="Crocus"
df['label'].iloc[400:480]="Iris"
df['label'].iloc[480:560]="Tigerlily"
df['label'].iloc[560:640]="Tulip"
df['label'].iloc[640:720]="Fritillary"
df['label'].iloc[720:800]="Sunflower"
df['label'].iloc[800:880]="Daisy"
df['label'].iloc[880:1040]="Dandelion"
df['label'].iloc[1040:1120]="Cowslip"
df['label'].iloc[1120:1200]="Buttercup"
df['label'].iloc[1200:1280]="Windflower"
df['label'].iloc[1280:1360]="Pansy"

df

df['label'].value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, df['label'], test_size=0.2, random_state=0, stratify=df['label'])

X_train.shape,X_test.shape, y_train.shape, y_test.shape

X_train['label'].value_counts()

X_test['label'].value_counts()

file = open('/content/drive/MyDrive/DIP/DIP_flowers/data/img.pkl', 'rb')
# dump information to that file
data_gray=pickle.load(file)
# close the file
file.close()
X_train_img_arr_gray,X_test_img_arr_gray = train_test_split(data_gray, test_size=0.2, random_state=0, stratify=df['label'])

file = open('/content/drive/MyDrive/DIP/DIP_flowers/data/all_img_color.pkl', 'rb')
# dump information to that file
data=pickle.load(file)
# close the file
file.close()

X_train_img_arr,X_test_img_arr = train_test_split(data, test_size=0.2, random_state=0, stratify=df['label'])

print(len(X_train_img_arr_gray))
print(len(X_test_img_arr_gray))
print(len(X_train_img_arr))
print(len(X_test_img_arr))

# X_train_img_arr=[]
# for i in X_train['img']: 
#   img = cv2.imread(i)
#   # print(img)
#   # print(img.shape)
#   X_train_img_arr.append(img)

# X_train_img_arr_gray=[]
# for i in X_train_img_arr: 
#   g=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
#   X_train_img_arr_gray.append(g)

# X_test_img_arr=[]
# for i in X_test['img']: 
#   img = cv2.imread(i)
#   X_train_img_arr.append(img)

# X_test_img_arr_gray=[]
# for i in X_test_img_arr:
#   g=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
#   X_test_img_arr_gray.append(g)

#Replace this code with proper image loading code from gdrive as following
#Load images as tuples (img, label)
#split into train test pair after shuffling
#trainX, trainY, testX, testY is obtained

#_________________________________________________#

# imgs = []
# for i in range(1,11):
#   name = 'image_{:04d}.jpg'.format(i)
#   im = cv2.imread(name,0)
#   #im = im.astype('uint8')
#   imgs.append((im, 0))

# for i in range(89,103):
#   im = cv2.imread('image_{:04d}.jpg'.format(i),0)
#   #im = im.astype('uint8')
#   imgs.append((im, 1))

# random.shuffle(imgs)
# train = list(zip(*imgs[:18]))
# trainX = train[0]
# trainY = train[1]

# test = list(zip(*imgs[18:]))
# testX = test[0]
# testY = test[1]

"""##Shape based Feature Extraction"""

def FeatureExtractor (imgs, extractor):
  imgsDescriptors = []
  allDescriptors = []
  for img in imgs:
    keypoint, descriptor = extractor.detectAndCompute(img,None)
    # print(descriptor)
    imgsDescriptors.append(descriptor)
    allDescriptors.extend(descriptor)

  return (allDescriptors, imgsDescriptors)



def normalize(hist):
  csum = np.sum(hist, axis = 1).reshape(-1,1)
  # print(csum)
  hist = hist / csum
  # print (hist.shape, hist[0].shape)
  # print(np.sum(hist, axis = 1))
  return list(hist)

# def load_img(xtrain):
#   if(len(images)!=0)
#     images=color_read(xtrain)
#   img_arr=gray_read(images)
#   # img_arr_train=[]
#   # for i in xtrain['img'][:20]: 
#   #     path = i
#   #     img = cv2.imread(path,0)
#   #     # print(img)
#   #     # print(img.shape)
#   #     img_arr_train.append(img)
#   return img_arr

def CreateVocabulary (k, descriptors):  
  #convert descriptors to float?
  vocabulary, distortion = kmeans(descriptors, k, 1)
  return vocabulary

def ComputeHistogram (imgsDescriptors, vocabulary, k):
  histograms = []
  for imgDescriptors in imgsDescriptors:
    histogram = np.zeros(k)
    words, dist = vq(imgDescriptors, vocabulary)
    for w in words:
      histogram[w] += 1
    # histogram = histogram.reshape(-1,1)
    histograms.append(histogram)
 
  return histograms

def Predict (testHistograms, trainHistograms, trainY):
  testY = []
  for test in testHistograms:
    #use eucledian dist for now, change to chi square later
    dists = np.square(test[np.newaxis,:] - trainHistograms).sum(axis=1)
    mostLike = np.argmin(dists)
    testY.append(trainY.iloc[mostLike])
  return testY

sift = cv2.SIFT_create()

# print(images)
(allDescriptors, imgsDescriptorsTrain) = FeatureExtractor(X_train_img_arr_gray, sift)


# -----------------------Pickle code--------------------------
# open a file, where you ant to store the data
file = open('Shape_allDescriptors.pkl', 'wb')
# dump information to that file
pickle.dump(allDescriptors, file)
# close the file
file.close()

file = open('Shape_imgsDescriptionTrain.pkl', 'wb')
# dump information to that file
pickle.dump(imgsDescriptorsTrain, file)
# close the file
file.close()
# ---------------------------------------------------------------

k = 300
vocabulary = CreateVocabulary(k, allDescriptors)
histTrain = ComputeHistogram(imgsDescriptorsTrain, vocabulary, k)
histTrain=normalize(np.array(histTrain))


(ignore, imgsDescriptorsTest) = FeatureExtractor(X_test_img_arr_gray, sift)
histTest = ComputeHistogram(imgsDescriptorsTest, vocabulary, k)
histTest=normalize(np.array(histTest))


# -----------------------Pickle code--------------------------
# open a file, where you ant to store the data
file = open('Shape_hist_train.pkl', 'wb')
# dump information to that file
pickle.dump(histTrain, file)
# close the file
file.close()

file = open('Shape_hist_test.pkl', 'wb')
# dump information to that file
pickle.dump(histTest, file)
# close the file
file.close()
# ---------------------------------------------------------------

pred_y=Predict(histTest, histTrain, y_train)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred_y))

#Testing on single image
# im = cv2.imread('image_0001.jpg',0)
# sift = cv2.SIFT_create()
# (allDescriptors, imgsDescriptors) = FeatureExtractor([im], sift)
# k = 20
# vocabulary = CreateVocabulary(k, allDescriptors)
# histograms = ComputeHistogram(imgsDescriptors, vocabulary, k)

# def FeatureExtractor (classToImgs, extractor):
#   #inputs images in the form {classLabel: [images]}
#   #outputs ( [all descriptors from all the images], {classLabel:[keypoints x 128 ndarray for each classLabel image]} )
#   classToImgDescriptors = {}
#   allDescriptors = []

#   for classLabel,images in classToImgs.items():
#     classToImgDescriptors[classLabel] = []
#     for img in images:
#       keypoint, descriptor = extractor.detectAndCompute(img,None)
#       classToImgDescriptors[classLabel].append(descriptor)
#       allDescriptors.extend(descriptor)

#   return (allDescriptors, classToImgDescriptors)



# def ComputeHistogram (classToImgDescriptors,vocabulary, k):
#   classToImageHistograms = {}
#   for classes, imgs in classToImgDescriptors.items():
#     classToImageHistograms[classes] = []
#     for img in imgs:
#       histogram = np.zeros(k)
#       words, dist = vq(img, vocabulary)
#       for w in words:
#         histogram[w] += 1
#       classToImageHistograms[classes].append(histogram)

