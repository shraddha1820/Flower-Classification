
"""## Color Based Extraction:

"""

def FeatureExtractor (imgs):
  imgsDescriptors = []
  allDescriptors = []
  for img in imgs:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('double')
    # print(list(hsv_img[:,:,0]))
    # print(hsv_img[:,:,0].flatten().tolist())
    # print(hsv_img[:,:].flatten().tolist())
    x =rfn.unstructured_to_structured(hsv_img)
    # print(x.flatten().tolist()[0:5])
    x=x.flatten().tolist()
    imgsDescriptors.append(x)
    allDescriptors.extend(x)
  return (allDescriptors, imgsDescriptors)

def Resize(imgs,dim=250):
  new_imgs=[]
  for i in imgs:
    x=cv2.resize(i, (), interpolation = cv2.INTER_CUBIC)
    new_imgs.append(x)
  return new_imgs



def CreateVocabulary (k, descriptors):  
  #convert descriptors to float?
  # print(descriptors)
  vocabulary=MiniBatchKMeans(n_clusters=k, batch_size = 6).fit(descriptors)
  # vocabulary, distortion = kmeans(descriptors, k, 1, check_finite=False)
  # vocabulary, distortion = kmeans(descriptors, k, 1)
  return vocabulary.cluster_centers_
  # return vocabulary

def ComputeHistogram (imgsDescriptors, vocabulary, k):
  histograms = []
  for imgDescriptors in imgsDescriptors:
    histogram = np.zeros(k)
    words, dist = vq(imgDescriptors, vocabulary)
    for w in words:
      histogram[w] += 1
    # histogram = histogram.reshape(-1)
    histograms.append(histogram)
  return histograms

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

from skimage.color import rgb2hsv

# X_train_img_arr=[]
# for i in X_train['img'][:1]: 
#   img = cv2.imread(i)
#   # print(img)
#   # print(img.shape)
#   X_train_img_arr.append(img)
print(X_train_img_arr[0].shape)
X_train_img_arr=Resize(X_train_img_arr[0:10])
print(X_train_img_arr[0].shape)
# allDescriptors_train, imgsDescriptors_train=FeatureExtractor(X_train_img_arr[0:500])


# # -----------------------Pickle code--------------------------
# # open a file, where you ant to store the data
# file = open('Color_All_Descriptors.pkl', 'wb')
# # dump information to that file
# pickle.dump(allDescriptors_train, file)
# # close the file
# file.close()

# file = open('Color_Image_Descriptors.pkl', 'wb')
# # dump information to that file
# pickle.dump(imgsDescriptors_train, file)
# # close the file
# file.close()
# # ---------------------------------------------------------------




# k = 300
# print("Features Extracted train")
# vocabulary = CreateVocabulary(k, allDescriptors_train)
# print("Vocabulary Created train")
# hist_train = ComputeHistogram(imgsDescriptors_train, vocabulary, k)
# hist_train=normalize(np.array(hist_train))
# # print("Histogram ban gaya train")
# # # print(hist_train)


# # -----------------------Pickle code--------------------------
# # open a file, where you ant to store the data
# file = open('Color_AllDescriptorsTrain', 'wb')
# # dump information to that file
# pickle.dump(allDescriptors_train, file)
# # close the file
# file.close()

# file = open('Color_ImgDescriptorsTrain', 'wb')
# # dump information to that file
# pickle.dump(imgsDescriptors_train, file)
# # close the file
# file.close()
# # file = open('important', 'rb') # dump information to that file 
# # data = pickle.load(file)       # close the file 
# # file.close()
# # ---------------------------------------------------------------



# allDescriptors_test, imgsDescriptors_test=FeatureExtractor(X_test_img_arr)
# hist_test = ComputeHistogram(imgsDescriptors_test, vocabulary, k)
# hist_test=normalize(np.array(hist_test))
# # for i in hist_train:
# #   print(i)
# # print(type(hist_train))



# # -----------------------Pickle code--------------------------

# file = open('Color_AllDescriptorsTest', 'wb')
# # dump information to that file
# pickle.dump(allDescriptors_test, file)
# # close the file
# file.close()

# file = open('Color_ImgDescriptorsTest', 'wb')
# # dump information to that file
# pickle.dump(imgsDescriptors_test, file)
# # close the file
# file.close()


# # open a file, where you ant to store the data
# file = open('Color_Histogram_train.pkl', 'wb')
# # dump information to that file
# pickle.dump(hist_train, file)
# # close the file
# file.close()

# file = open('Color_Histogram_test.pkl', 'wb')
# # dump information to that file
# pickle.dump(hist_test, file)
# # close the file
# file.close()
# # file = open('Color_Histogram_test.pkl', 'wb')
# # # dump information to that file
# # pickle.dump(hist_test, file)
# # # close the file
# # file.close()


# # ---------------------------------------------------------------

pred_y=Predict(hist_test,hist_train,y_train[:5])
print(pred_y)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test[0:20], pred_y))

def Predict (testHistogramsS, trainHistogramsS, testHistogramsT, trainHistogramsT,testHistogramsC, trainHistogramsC, trainY, a1=1, a2=1, a3=0.4):
  testY = []
  for testS, testT, testC in zip(testHistogramsS,testHistogramsT, testHistogramsC):
    #use eucledian dist for now, change to chi squaallDescriptors, imgsDescriptorsre later
    distsS = np.square(testS[np.newaxis,:] - trainHistogramsS).sum(axis=1)  
    distsT = np.square(testT[np.newaxis,:] - trainHistogramsT).sum(axis=1)  
    distsC = np.square(testC[np.newaxis,:] - trainHistogramsC).sum(axis=1) 
    dists = a1 *  distsS + a2* distsT + a3 * distsC
    mostLike = np.argmin(dists)
    # print(mostLike)
    # print("---",trainY.iloc[mostLike])
    testY.append(trainY.iloc[mostLike])
  return testY

