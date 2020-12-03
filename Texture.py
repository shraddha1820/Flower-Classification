

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

# def load_image():
#     strr = "Datasets/*.jpg"
#     for file in glob.glob(strr):
#         img=np.asarray(plt.imread(file))
#         arr.append(img)
#     return arr

def preprocessing(df):
    arr_prep=[]
    for i in range(len(df)):
        img=cv2.cvtColor(df['img'][i],0)
        arr_prep.append(img)
    return arr_prep

def extractLBP(img):
    lbp = local_binary_pattern(img, 24,3, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27),range=(0, 26))                   
    hist = hist.astype("float")                         
    hist /= (hist.sum() + (1e-7))
    return lbp,hist    

def featureExtraction(arr):
    arr_feature=[]
    vector_feature=[]
    for i in range(len(arr)):
        lb,vektor = extractLBP(arr[i])
        # print(lb.shape)
        arr_feature.append(lb)
        vector_feature.append(vektor)
    return arr_feature, vector_feature


def normalize(hist):
  csum = np.sum(hist, axis = 1).reshape(-1,1)
  # print(csum)
  hist = hist / csum
  # print (hist.shape, hist[0].shape)
  # print(np.sum(hist, axis = 1))
  return list(hist)

def Predict (testHistograms, trainHistograms, trainY):
  testY = []
  for test in testHistograms:
    #use eucledian dist for now, change to chi squaallDescriptors, imgsDescriptorsre later
    dists = np.square(test[np.newaxis,:] - trainHistograms).sum(axis=1)       #using normal distance
    # dist = cv2.compareHist(test,train, cv2.HISTCMP_CHISQR)
    mostLike = np.argmin(dists)
    # print(mostLike)
    # print("---",trainY.iloc[mostLike])
    testY.append(trainY.iloc[mostLike])
  return testY

dataExtracted,vector_hist_train = featureExtraction(X_train_img_arr_gray)
vector_hist_train=normalize(vector_hist_train)

# arr_test= []
# arr_test=load_img(X_test)    #shape feature's load function used 
dataExtracted,vector_hist_test = featureExtraction(X_test_img_arr_gray)
vector_hist_test=normalize(vector_hist_test)


# -----------------------Pickle code--------------------------
# open a file, where you ant to store the data
file = open('Texture_hist_train.pkl', 'wb')
# dump information to that file
pickle.dump(vector_hist_train, file)
# close the file
file.close()

file = open('Texture_hist_test.pkl', 'wb')
# dump information to that file
pickle.dump(vector_hist_test, file)
# close the file
file.close()
# ---------------------------------------------------------------

print('Extraction Result')
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(4,2,1)
ax1.set_title('Before')
ax1.set_axis_off()
ax1.imshow(arr_prep[0])

ax2 = fig.add_subplot(4,2,2)
ax2.set_title('After')
ax2.set_axis_off()
ax2.imshow(dataExtracted[2],cmap=plt.cm.gray)
plt.show()

# print("Vector of Image 1 :",vector[0])
# print()
# print("Vector of Image 2 :",vector[1])
# print()
# print("Vector of Image 3 :",vector[2])

pred_y=Predict(vector_hist_test,vector_hist_train,y_train)
print(pred_y)
print(y_test[0:20])

len(pred_y)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred_y))