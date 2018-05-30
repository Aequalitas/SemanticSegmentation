# dataset provider for machine learning frameworks

import os 
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import sys 
import json
from time import sleep


class Data:

    #self.config = None
    
    # path to the training and test images
    #self.pathImages = None
    # actual train and test data or file names of these
    # when serialized = false
    #self.imageData = None
    
    # loads the all file names or the serialized numpy object
    def loadDataset(self):
        self.pathImages = {
            "train": self.config["path"],
            "label": self.config["path"]
        }

        if self.config["preProcessedPath"] != "":
            self.pathImages["train"] += self.config["preProcessedPath"]+self.config["images"]
            self.pathImages["label"] += self.config["preProcessedPath"]+self.config["labels"]
        else:
            self.pathImages["train"] += self.config["images"]
            self.pathImages["label"] += self.config["labels"]
            
        if not self.config["serializedObject"]:
                self.imageData = {
                    "train": list(filter(lambda i:i != None, trainData)),
                    "label": list(filter(lambda i:i != None, labelData))
                }

            else:

                self.imageData = {
                    "train": os.listdir(self.pathImages["train"]),
                    "label": os.listdir(self.pathImages["label"])
                }
        else:
            self.imageData = np.load(self.config["path"]+self.config["fileName"])
            print("Finished loading dataset...")


    # string configPath - path of the json file which describes the dataset
    def __init__(self, configPath):

        try:
            self.config = json.load(open(configPath))    
        except:
            raise "Wrong path for data config file given!"

        self.loadDataset()

    # gets a value from the config file with its given name
    def getConfig(self, name):
        return self.config[name]

    # calculates the class weights for the current dataset
    # string weightType - how the class weights are calculated
    def getClassWeights(self, weightType):
   
        if not weightType in ["Freq", "MedianFreq", "1x", "2x", "division"]:
            raise ValueError("Wrong weights calc type given! Valid arguments are [Freq, MedianFreq, 1x, 2x, division]")


        # get class weights because of imbalanced dataset (e.g. a lot of road and buildings)
        print("Calculate class weights...")

        # count all labels
        PART = 10
        labels = np.zeros((self.config["y"], self.config["x"]))
        classCount = np.zeros(self.config["classes"])

        for i in range(int(self.config["size"]/PART)):
            labels = getImg(i, "label").flatten().astype("int")
            classCount[labels] += 1

            if i % 100 == 99:
                print("Label image ",i+1,"/", ds["size"]/PART)
        
        #choose class weights type
        #Frequency
        if weightType == "Freq":
            classWeights =  np.median(classCount) / classCount
        #Median Frequency
        elif weightType == "MedianFreq":
            classWeights = np.median(np.median(classCount) / classCount)/(np.median(classCount) / classCount)
        # Simple Total/ClassCount
        elif weightType == "1x":
            classWeights = 1 - (classCount / classCount.sum()*1)
        # Simple Total/ClassCount doubled effect
        elif weightType == "2x":
            classWeights = 1 - (classCount / classCount.sum()*2)
        # Simple Total/ClassCount divided by Minimum
        elif weightType == "division":
            classWeights = classCount.sum() / classCount
            #divide with minimum
            classWeights[classWeights == 1] = 999999
            classWeights /= classWeights.min()
            classWeights[classWeights == classWeights.max()] = 1
        else:
            raise ValueError("Wrong weights calc type given! Valid arguments are [Freq, MedianFreq, 1x, 2x, division]")

        # eliminate inf values
        classWeights[(classWeights == np.inf)] = 1
        
        print(classCount.shape)
        print(classCount)
        print(classWeights.shape)
        print(classWeights)

        return classWeights
    
    # get the whole dataset as an numpy object
    # bool balanced returns an dataset that has even amount of every class set by the smallest class -> extreme balancing
    def getDataset(self, balanced=False):
        
        trainData = np.empty((self.config["size"], self.config["x"], self.config["y"], self.config["imageChannels"]), dtype=np.float64)
        labelData = np.empty((self.config["size"], self.config["classes"]), dtype=np.uint8)

        # can not select ranges from with [:,0], only [i,0] works
        # therefore seperate lists has to be made
        for i in range(0, self.config["size"]):
            if self.config["imageChannels"] == 3:
                img = np.expand_dims(cv2.resize(self.imageData[i,0], (self.config["x"], self.config["y"]), interpolation=cv2.INTER_NEAREST), axis=3)
                trainData[i] = np.append(np.append(img, img, axis=2), img, axis=2)
            else:
                trainData[i] = self.imageData[i,0]
            
            labelData[i] = self.imageData[i,1]
        
        if balanced:
            classes = np.argmax(labelData, axis=1)
            smallestClassCount = np.bincount(classes).min()
            smallestClass = np.argmin(np.bincount(classes))
            print("Smallest Class with ", smallestClassCount, " elements is class ", smallestClass, " from ", np.bincount(np.argmax(labelData, axis=1)))
            
            classIndices = [[0] * 10000 for i in range(self.config["classes"])]#np.array((self.config["classes"], self.config["size"]))

            for classNr in range(self.config["classes"]):
                classIndices[classNr] = [el[:smallestClassCount] for el in np.where((classes == classNr))]

            # this part is way to slow: 10 minutes for 500 elements
            # stays commented for educational purposes
            # for idx, x in enumerate(np.argmax(labelData, axis=1)):
            #     print(idx)
            #     if classCounts[x] > smallestClass:
            #        np.delete(labelData, idx)
            #        np.delete(trainData, idx)
            #        classCounts = np.bincount(np.argmax(labelData, axis=1))

            np.random.shuffle(classIndices)
            classIndices = np.asarray(classIndices)
            classIndices = classIndices.flatten()
            trainData = trainData[classIndices]
            labelData = labelData[classIndices]

        return np.array(trainData), np.array(labelData)
    
    
    # reads an image and pre-processes it for training/testing
    # int i - index for the image
    # string type - wether it is a train or label image
    def getImage(self, i, type):
        imageName = self.imageData[type][i]
        
        if self.config["preProcessedPath"] != "":
            if type == "label":
                return cv2.imread(self.pathImages[type]+imageName)[:,:,0]
            else:
                return cv2.imread(self.pathImages[type]+imageName)

        img = cv2.imread(self.pathImages[type]+imageName)

        if self.config["downsize"]:
            img = cv2.resize(img, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if type == "label":
            for rgbIdx, rgbV in enumerate(self.config["ClassToRGB"]):
                labelMask = (img == rgbV)
                img[labelMask] = rgbIdx

            return img[:,:,0]

        elif type == "train":
            #  standarsize train values to a range of [-1,1]
            if img.std() != None :
                return (img - img.mean()) / img.std()
            else:
                print("NONE IMG")
                return (img - img.mean()) / 1

    # returns an iterator which provides on every call one batch
    # int bsize - batch size to be returned
    def getNextBatchTrain(self, bsize):
        batchCount = list(range(int(self.config["trainsize"]*(int(self.config["size"])/bsize))))
        np.random.shuffle(batchCount)
        for x in batchCount:
            data = [[],[]]
            for i in range(bsize):
                #label
                data[0].append(self.imageData[x+i, 1] if self.config["serializedObject"] else self.getImage(x+i, "label"))
                #train
                data[1].append(self.imageData[x+i, 0] if self.config["serializedObject"] else self.getImage(x+i, "train"))

            yield np.asarray(data[0]), np.asarray(data[1])

    # returns a batch of test data
    # int testsize - size of the to be returned test data
    def getNextBatchTest(self, testsize):
        # test is considered one batch
        data = [[],[]]
        tS = testsize#self.config["trainsize"] * self.imageData[0].size
        batchCount = list(range(testsize))
        for x in batchCount:
            #label
            data[0].append(self.imageData[x+tS, 1] if self.config["serializedObject"] else self.getImage(x+tS, "label"))
            #train
            data[1].append(self.imageData[x+tS, 0] if self.config["serializedObject"] else self.getImage(x+tS, "train"))
                
        return np.asarray(data[0]), np.asarray(data[1])
    
    # returns a generator of test data
    # int testsize - size of the to be returned test data
    def getNextBatchTestGenerator(self, testsize):
        # test is considered one batch
        data = [[],[]]
        tS = testsize#self.config["trainsize"] * self.imageData[0].size
        batchCount = list(range(testsize))
        for x in batchCount:
            #label
            data[0].append(self.imageData[x+tS, 1] if self.config["serializedObject"] else getImage(x+tS, "label"))
            #train
            data[1].append(self.imageData[x+tS, 0] if self.config["serializedObject"] else getImage(x+tS, "train"))
                
        yield np.asarray(data[0]), np.asarray(data[1])