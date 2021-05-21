# TEXT DETECTION CNN....

# TRINING SCRIPT

import cv2
import numpy as np
import os

######################### SETTINGS

path = "Text Data"

############################

# Create the data as list

myList = os.listdir(path) #list directry
print(myList)

noOfClasses = len(myList)
print("Total no of classes Detected",noOfClasses)

images = []  # make list of the images in the folder one by one
classNo = []
# To put all images in a list for that we need iteration so we using For loop

# the images in folder 0 to 9 we make the images as list
print("Importing Classes....")
for x in range(0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32)) #resize will more efficien in training process
        images.append(curImg)
        classNo.append(x)
    print(x,end =" ") # to print in horizontal line

#### COVERT IT INTO NUMPY ARRAY ###

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)

#################### SPLITING OF DATA ######################
print("Spliting Of Data")