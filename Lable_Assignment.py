import cv2
import numpy as np

import os

# Path of working folder on Disk
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
Lables =[]
Images = load_images_from_folder("/home/ahmedhasan/PycharmProjects/FYP/digitRecognition/DataSet")
for i in range(len(Images)):

    img = Images[i]
    cv2.imshow("Resulting Image with Rectangular ROIs", img)
    cv2.waitKey()
    x = int(input("Enter a number: "))
    Lables.append(x)
    cv2.imwrite("/home/ahmedhasan/PycharmProjects/FYP/digitRecognition/FinalDataSet/" + str(x) + ".jpeg", img)
    cv2.destroyAllWindows()

Lab = np.array(Lables)
np.save("DigitsLable.npy",Lab)



