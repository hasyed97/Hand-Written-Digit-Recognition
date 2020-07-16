import os
import cv2 as cv


max_value = 255
max_type =  255
max_binary_value = 255
trackbar_type = 'Type:Value'
trackbar_value = 'Value'
window_name = 'Threshold Demo'

def nothing(x):
    pass
def round_up_to_odd(x):
    return 2 * ((int)(x / 2.0) ) + 1;

def Threshold_Demo(val):
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)

    dst = cv.adaptiveThreshold(src_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,round_up_to_odd(threshold_type),round_up_to_odd(threshold_value))
    cv.imshow(window_name, dst)
    callbackButton(dst,index)

def callbackButton(greyimg,ind):
    cv.imwrite("/home/ahmedhasan/PycharmProjects/FYP/digitRecognition/Test/imagegrey1191221"+str(ind)+".png", greyimg)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def crop_images(path):
    im = cv.imread(path)
    k = 0
    imgnum=1
    while k != 113:
        # Select ROI
        r = cv.selectROI(im)
        # Crop image
        imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Display cropped image
        cv.imwrite("/home/ahmedhasan/PycharmProjects/FYP/digitRecognition/te/Image"+str(imgnum)+".png", imCrop)
        # im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = 255
        k = cv.waitKey()
        cv.destroyAllWindows()
        imgnum = imgnum+1
crop_images("WhatsApp Image 2020-01-03 at 11.38.33 PM.jpeg")
Images = load_images_from_folder("/home/ahmedhasan/PycharmProjects/FYP/digitRecognition/te")
# Read the input image# for index in range(len(Images)):
for index in range(len(Images)):
    src = Images[index]
    if src is None:
        print('Could not open or find the image: ')
        exit(0)
    # Convert the image to Gray
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.namedWindow(window_name)
    cv.createTrackbar(trackbar_type, window_name , 17, max_type, Threshold_Demo)
    # Create Trackbar to choose Threshold value
    cv.createTrackbar(trackbar_value, window_name , 5, max_value, Threshold_Demo)
    # Call the function to initialize
    Threshold_Demo(0)

    # Wait until user finishes program
    cv.waitKey()
    cv.destroyAllWindows()
