#Import the modules

import timeit
code_to_test = """
import os
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

# Load the classifier
# Un comment to predict with SVM Model
# clf = joblib.load('SVM_Train_Model.pkl')
# Comment to predict with SVM Model
clf = load_model('MNIST_CNN_Model.h5')
Images = load_images_from_folder("/home/ahmedhasan/PycharmProjects/FYP/digitRecognition/DataSet")
# Read the input image# for index in range(len(Images)):
lable = np.load('DigitsLable.npy')
Acc = 0
for index in range(len(Images)):
    im = Images[index]
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)
    # Find contours in the image
    ctrs,hierachy =  cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours according to position on image
    ctrs.sort(key=lambda x: get_contour_precedence(x, im.shape[2]))

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    # For each rectangular region, convert it to np.array and expand dimensions
    # the array use CNN to predict
    res=""
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        try:
            # Adding padding to center the image 
            roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT)
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except:
            continue
        # Un comment to predict with SVM Model
        
        # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        
        # Comment to predict with SVM Model
        
        x = img_to_array(roi)
        x = np.expand_dims(x, axis=0)
        nbr = clf.predict_classes(x)
        
        # Put predicted digit around the original 
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        res=res+str(nbr[0])
    pridicted = int(res)
    if(pridicted==lable[index]):
        Acc=Acc+1
    
    # UnComment to check the prediction on the image
    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
    # cv2.waitKey()
# Show the Test Accuracy on console
print("Accuracy : "+str(Acc/len(lable)*100))
"""
elapsed_time = timeit.timeit(code_to_test, number=100)/100
print("Execution Time : "+str(elapsed_time))