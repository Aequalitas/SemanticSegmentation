# Author: https://github.com/Aequalitas
# This class provides some utility functions to work with a dataset

import os 
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

import sys 
import random
import json
from time import sleep

class Data:

    # loads all the in the dataset config file specified file names or the serialized numpy object
    def loadDataset(self):
        # set dictonary for the different dataset splits and set initially the dataset path
        self.pathImages = {
            "train": self.config["path"],
            "trainLabel": self.config["path"],
            "test" : self.config["path"],
            "testLabel": self.config["path"],
            "validation": self.config["path"],
            "validationLabel": self.config["path"]
        }
        
        # append the given train or label path
        self.pathImages["train"] += self.config["images"]
        self.pathImages["trainLabel"] += self.config["labels"]
        self.pathImages["test"] += self.config["images"]
        self.pathImages["testLabel"] += self.config["labels"]
        self.pathImages["validation"] += self.config["images"]
        self.pathImages["validationLabel"] += self.config["labels"]
        
        if not self.config["serializedObject"]:
            # with os.listdir() read the file names in the directories
            trainDataFiles = os.listdir(self.pathImages["train"])
            trainLabelDataFiles = os.listdir(self.pathImages["trainLabel"])
            
            # sort file names because os.listdir does extract them in arbitrary order
            trainDataFiles.sort()
            trainLabelDataFiles.sort()
            
            # count the amount of the file names that also sets the training size
            trainElements = int(self.config["trainSize"]*len(trainDataFiles))
            testElements = int(self.config["testSize"]*len(trainDataFiles))
            
            # remove n elements in order for a complete last batch with axis = 0 => batchSize
            trainElements -= trainElements % self.config["batchSize"]
            testElements -= testElements % self.config["batchSize"]
            
            # shuffle the file names for creating a balanced training experience
            # same random seed to be able to compare results with other training sessions
            # here the sum of chars in the dataset name. Calc function taken from: https://codereview.stackexchange.com/q/13863
            random.seed(sum(ord(c) - 64 for c in self.config["name"]))
            randomIndices = np.arange(len(trainDataFiles), dtype=np.int32)
            random.shuffle(randomIndices)
            trainDataFiles = np.take(trainDataFiles, randomIndices)
            trainLabelDataFiles = np.take(trainLabelDataFiles, randomIndices)
            
            # set the given dataset split whith their element by simple numpy indexing
            self.imageData = {
                "train": trainDataFiles[:trainElements],
                "trainLabel": trainLabelDataFiles[:trainElements],
                "test": trainDataFiles[trainElements:trainElements+testElements],
                "testLabel": trainLabelDataFiles[trainElements:trainElements+testElements],
                "validation": trainDataFiles[trainElements+testElements if testElements > 0 else trainElements:],
                "validationLabel": trainLabelDataFiles[trainElements+testElements if testElements > 0 else trainElements:]
            }
            # set the dataset splits sizes
            self.config["trainSize"] = len(self.imageData["train"])
            self.config["testSize"] = len(self.imageData["test"])
            self.config["validationSize"] = len(self.imageData["validation"])
            print("trainSize: ", self.config["trainSize"], " Testsize: ", self.config["testSize"], "Validationsize: ", self.config["validationSize"])

        # else read serialized numpy object
        else:
            self.imageData = np.load(self.config["path"]+self.config["fileName"]+".npy")
            self.config["trainSize"] = len(self.imageData.item().get("train"))
            self.config["testSize"] = len(self.imageData.item().get("test"))
            self.config["validationSize"] = len(self.imageData.item().get("validation"))
            print("Finished loading dataset...")


    # string configPath - path of the json file which describes the dataset
    def __init__(self, configPath):

        try:
            print(configPath)
            self.config = json.load(open(configPath))    
        except:
            raise "Wrong path for data config file given!"

    
        self.loadDataset()

    # gets a value from the config file with its given name
    def getConfig(self, name):
        return self.config[name]

    # calculates the class weights for the current dataset
    # string weightType - how the class weights are calculated
    # [trainData, labelData] dataSet - if there is another dataset than self.getDataset() eg. balanced
    def getClassWeights(self, weightType, dataSet=None):
   
        if not weightType in ["Freq", "MedianFreq", "1x", "2x", "division", "relativeToMin", "quantile"]:
            raise ValueError("Wrong weights calc type given! Valid arguments are [Freq, MedianFreq, 1x, 2x, division, relativeToMin, quantile]")

        # get class weights because of imbalanced dataset (e.g. a lot of road and buildings)
        print("Calculate class ",weightType ," weights...")

        # only calculate the weights from a specific split of the dataset. For performance reasons
        # PART = 1 would be the total dataset
        PART = 10
        classCount = np.zeros(self.config["classes"])
        # count all the classes in every given mask image
        for i in range(int(self.config["trainSize"]/PART)):
            labelImg = self.getImage(i, "trainLabel").flatten()
            labelClassCount = np.bincount(labelImg, minlength=self.config["classes"])
            classCount += labelClassCount
            
            if i % int(1000/PART) == 0:
                print("Label image ",i,"/", self.config["trainSize"]/PART)

        print("Class count: ", classCount.shape, classCount)

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
        # all weights are relative to the smallest class which is assigned the 1.0. Minimal assigned value is 0.1                     
        elif weightType == "relativeToMin":
            classWeights = classCount.min() / classCount
            print("Class weights: ", classWeights.shape, classWeights)
            classWeights[(classWeights < 0.1)] *= 10 
        # using the quantile transformer of sklearn all the weights are distributed in 0-0.9999. Minimal assigned value is 0.1
        elif weightType == "quantile":
            from sklearn.preprocessing.data import QuantileTransformer
            _scaler = QuantileTransformer()
            classCount = np.expand_dims(classCount, axis=1)
            classWeights = _scaler.fit_transform(classCount)
            classWeights = np.around(classWeights, decimals=4)
            classWeights = np.squeeze(classWeights)
            classWeights = 1 - classWeights
            classWeights[(classWeights < 0.1)] = 0.1
            
        else:
            raise ValueError("Wrong weights calc type given! Valid arguments are [Freq, MedianFreq, 1x, 2x, division, relativeToMin, quantile]")

        # eliminate inf values
        classWeights[(classWeights == np.inf)] = 1
        print("Class weights: ", classWeights.shape, classWeights)
        np.save("classWeights"+str(self.config["x"])+str(self.config["y"])+self.config["name"], classWeights)
    
    # get the whole dataset as an numpy object for the "serialize" MODE
    # bool balanced returns an train dataset that has even amount of every class set by the smallest class -> extreme balancing
    # bool flipV whether the dataset is doubled by flipping all images vertically
    def getDataset(self, balanced=False, flipV=False):
        
        dataSet = {
            "train": None,
            "trainLabel": None,
            "test": None,
            "testLabel": None,
            "validation": None,
            "validationLabel": None
        }
        
        # fill the sets with the appropiate pre-processed images
        for set in ["train", "test", "validation"]:
            size = self.config[set+"Size"]
            dataSet[set] = np.array([self.getImage(x, set) for x in range(size)])
            dataSet[set+"Label"] = np.array([self.getImage(x, set+"Label") for x in range(size)])
                
        if balanced:
            classes = np.argmax(dataSet["label"], axis=1)
            smallestClassCount = np.bincount(classes).min()
            smallestClass = np.argmin(np.bincount(classes))
            print("Smallest Class with ", smallestClassCount, " elements is class ", smallestClass, " from ", np.bincount(np.argmax(dataSet["label"], axis=1)))
            classIndices = [[0] * 10000 for i in range(self.config["classes"])]
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
            dataSet["train"] = dataSet["train"][classIndices]
            dataSet["trainLabel"] = dataSet["trainLabel"][classIndices]
        
        # flip images vertically
        if flipV:
            for set in ["train", "test", "validation"]:
                dataSet[set] = np.append(dataSet[set], [np.fliplr(x) for x in dataSet[set]], axis=0)
                dataSet[set+"Label"] = np.append(dataSet[set+"Label"], [np.fliplr(x) for x in dataSet[set+"Label"]], axis=0)

        return dataSet
    
    # reads an image and pre-processes it for training/testing
    # string imageFilename name(s) of the current train batch
    # string labelFilename name(s) of the current label batch
    def getImageTuple(self, imageFilename, labelFilename):
        
        # with tensorflow

        #imgPath = tf.string_join([self.pathImages["train"], imageFilename])
        #imgString = tf.read_file(imgPath)
        #img = tf.image.decode_jpeg(imgString, channels=3)

        #labelImgPath = tf.string_join([self.pathImages["trainLabel"], labelFilename])
        #labelImgString = tf.read_file(labelImgPath)
        #labelImg = tf.image.decode_jpeg(labelImgString)

        #if self.config["downsize"]:
        #    img = tf.image.resize_images(img, [self.config["x"], self.config["y"]])
        #    labelImg = tf.image.resize_images(labelImg, [self.config["x"], self.config["y"]])

        # labels
        #classesImg = tf.zeros([self.config["y"] * self.config["x"]], tf.int32)
        # converting RGB values to their corresponding classes
        #for rgbIdx, rgbV in enumerate(self.config["ClassToRGB"]):
        #    rgbV = tf.constant(rgbV)
        #    labelImg = tf.cast(tf.reshape(labelImg, [self.config["y"] * self.config["x"]]), tf.int32)
            # way too slow, not comparable to numpys img[(img == rgbV).all(-1)] = rgbIdx
        #    label = tf.map_fn(lambda label: rgbIdx if tf.equal(label, rgbV) is not False else 0, labelImg)
            ##label = tf.SparseTensor(indices=tf.where(label), values=[rgbIdx], dense_shape=[data.config["x"] * data.config["y"]])
            ##label = tf.parse.to_dense(label)
        #    classesImg = tf.add(classesImg, label)

        #classesImg = tf.reshape(classesImg, [self.config["y"], self.config["x"]])

        #img = tf.image.per_image_standardization(img)
        #img = tf.reshape(img, [self.config["y"], self.config["x"], self.config["imageChannels"]])

        #return  img, classesImg

        # with numpy and opencv

        img = cv2.imread(self.pathImages["train"]+imageFilename.decode())
        labelImg = cv2.imread(self.pathImages["trainLabel"]+labelFilename.decode())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labelImg = cv2.cvtColor(labelImg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_AREA)
        if self.config["downsize"]: 
            labelImg = cv2.resize(labelImg, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_AREA)
        
        # assure that there are no conversion errors in binary datasets
              
        if self.config["classes"] <= 2:
            labelImg[(labelImg  >= 1).all(-1)] = [255,255,255]
            #labelImg[(labelImg  <= 127).all(-1)] = [0,0,0]
        
        #display(Image.fromarray(img, "RGB"))
        #display(Image.fromarray(labelImg, "RGB"))
        # transform the RGB values of the mask to the class numbers according to the list set in the dataset config file
        
        for rgbIdx, rgbV in enumerate(self.config["ClassToRGB"]):
            labelImg[(labelImg == rgbV).all(-1)] = rgbIdx
        
        
        # assure that there are no RGB values left by assigning them the black class as in the zero
        #labelImg[(labelImg >= self.config["classes"])] = 0
        
        
        # standardasize the train image for better training
        img = ((img - img.mean()) / img.std()).astype(np.float32)

        return img, labelImg[:,:,0].astype(np.int32)
