# dataset provider for classifications

import os 
import cv2
import numpy as np
from PIL import Image
import sys 
import json
from time import sleep
from IPython.display import display 


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
            "trainLabel": self.config["path"],
            "test" : self.config["path"],
            "testLabel": self.config["path"],
            "validation": self.config["path"],
            "validationLabel": self.config["path"]
        }

        if self.config["preProcessedPath"] != "":
            self.pathImages["train"] += self.config["preProcessedPath"]+self.config["images"]
            self.pathImages["trainLabel"] += self.config["preProcessedPath"]+self.config["labels"]
        else:
            self.pathImages["train"] += self.config["images"]
            self.pathImages["trainLabel"] += self.config["labels"]
            self.pathImages["test"] += self.config["images"]
            self.pathImages["testLabel"] += self.config["labels"]
            self.pathImages["validation"] += self.config["images"]
            self.pathImages["validationLabel"] += self.config["labels"]

        if not self.config["serializedObject"]:
            
            if self.config["name"] == "Seagrass":       
                jsonData = json.load(open(self.config["path"]+"train.json"))
                trainData = list(map(lambda i:os.path.basename(i["image"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                labelData = list(map(lambda i:os.path.basename(i["ground-truth"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))

                jsonData = json.load(open(self.config["path"]+"test.json"))
                testData = list(map(lambda i:os.path.basename(i["image"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                testLabelData = list(map(lambda i:os.path.basename(i["ground-truth"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))

                jsonData = json.load(open(self.config["path"]+"validate.json"))
                validateData = list(map(lambda i:os.path.basename(i["image"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                validateLabelData = list(map(lambda i:os.path.basename(i["ground-truth"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                
                self.imageData = {
                    "train": list(filter(lambda i:i != None, trainData)),
                    "trainLabel": list(filter(lambda i:i != None, labelData)),
                    "test": list(filter(lambda i:i != None, testData)),
                    "testLabel": list(filter(lambda i:i != None, testLabelData)),
                    "validation": list(filter(lambda i:i != None, validateData)),
                    "validationLabel": list(filter(lambda i:i != None, validateLabelData))
                }
            else:
                # sort data because os.listdir selects files in arbitrary order
                trainDataFiles = os.listdir(self.pathImages["train"])
                trainLabelDataFiles = os.listdir(self.pathImages["trainLabel"])
                trainDataFiles.sort()
                trainLabelDataFiles.sort()


                trainElements = int(self.config["trainSize"]*self.config["size"])
                testElements = int(self.config["testSize"]*self.config["size"])

                self.imageData = {
                    "train": trainDataFiles[:trainElements],
                    "trainLabel": trainLabelDataFiles[:trainElements],
                    "test": trainDataFiles[trainElements:trainElements+testElements],
                    "testLabel": trainLabelDataFiles[trainElements:trainElements+testElements],
                    "validation": trainDataFiles[trainElements+testElements if testElements > 0 else trainElements:],
                    "validationLabel": trainLabelDataFiles[trainElements+testElements if testElements > 0 else trainElements:],
                }

                
            self.config["trainSize"] = len(self.imageData["train"])
            self.config["testSize"] = len(self.imageData["test"])
            self.config["validationSize"] = len(self.imageData["validation"])

            print("trainSize: ", self.config["trainSize"], " Testsize: ", self.config["testSize"], "Validationsize: ", self.config["validationSize"])

        else:

            self.imageData = np.load(self.config["path"]+self.config["fileName"]+".npy")
            self.config["trainSize"] = len(self.imageData.item().get("train"))
            self.config["testSize"] = len(self.imageData.item().get("test"))
            self.config["validationSize"] = len(self.imageData.item().get("validation"))
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
    # [trainData, labelData] dataSet - if there is another dataset than self.getDataset() eg. balanced
    def getClassWeights(self, weightType, dataSet=None):
   
        if not weightType in ["Freq", "MedianFreq", "1x", "2x", "division"]:
            raise ValueError("Wrong weights calc type given! Valid arguments are [Freq, MedianFreq, 1x, 2x, division]")


        # get class weights because of imbalanced dataset (e.g. a lot of road and buildings)
        print("Calculate class weights...")

        # count all labels
        PART = 10
        labels = np.zeros((self.config["y"], self.config["x"]))
        classCount = np.zeros(self.config["classes"])

        for i in range(int(self.config["trainSize"]/PART)):
            labels = self.getImage(i, "trainLabel").flatten().astype("int")
            classCount[labels] += 1

            if i % 100 == 99:
                print("Label image ",i+1,"/", self.config["trainSize"]/PART)
        
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

        np.save("classWeights"+str(self.config["x"])+str(self.config["y"])+self.config["name"], classWeights)
    
    # get the whole dataset as an numpy object
    # bool balanced returns an train dataset that has even amount of every class set by the smallest class -> extreme balancing
    def getDataset(self, balanced=False, flipH=False):
        
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
            dataSet["train"] = dataSet["train"][classIndices]
            dataSet["trainLabel"] = dataSet["trainLabel"][classIndices]
        
        # flip images horizontally
        if flipH:
            for set in ["train", "test", "validation"]:
                dataSet[set] = np.append(dataSet[set], [np.fliplr(x) for x in dataSet[set]], axis=0)
                dataSet[set+"Label"] = np.append(dataSet[set+"Label"], [np.fliplr(x) for x in dataSet[set+"Label"]], axis=0)

        return dataSet
    
    
    def getImageTuple(self, imageFilename, labelFilename):
        img = cv2.imread(self.pathImages["train"]+imageFilename.decode())
        labelImg = cv2.imread(self.pathImages["trainLabel"]+labelFilename.decode())
        if self.config["downsize"]:
            img = cv2.resize(img, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_NEAREST)
            labelImg = cv2.resize(labelImg, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labelImg = cv2.cvtColor(labelImg, cv2.COLOR_BGR2RGB)

        # exterminate conversion errors by opencv
        #labelImg[(labelImg <= 127)] = 0
        #labelImg[(labelImg >= 128)] = 255
        
        labelImg = labelImg.astype(np.int32)

        for rgbIdx, rgbV in enumerate(self.config["ClassToRGB"]):
            labelImg[(labelImg == rgbV).all(-1)] = rgbIdx

        img = ((img - img.mean()) / img.std()).astype(np.float32)

        return img, labelImg[:,:,0]


    
    # reads an image and pre-processes it for training/testing
    # int i - index for the image
    def getImage(self, i, type):
        imageName = self.imageData[type][i]        
        
        if self.config["preProcessedPath"] != "":
            if type == "trainLabel":
                return cv2.imread(self.pathImages[type]+imageName)[:,:,0]
            else:
                return cv2.imread(self.pathImages[type]+imageName)

        img = cv2.imread(self.pathImages[type]+imageName)
        
        if self.config["downsize"]:
            img = cv2.resize(img, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_NEAREST)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # correct transformation errors of cv2
        
        #if type == "trainLabel":
        #    img[(img <= 127).all(-1)] = [0,0,0]
        #    img[(img >= 128).all(-1)] = [255,255,255]
        
        
        #print(self.pathImages[type]+imageName, img.mean())
        #display(Image.fromarray(img, "RGB"))
        
        

        if type in ["trainLabel", "testLabel", "validationLabel"]:
            
            for rgbIdx, rgbV in enumerate(self.config["ClassToRGB"]):
                img[(img == rgbV).all(-1)] = rgbIdx
            
            
            #print("Label: "+imageName+" ", img[:,:,0].max(), img[:,:,0].min(), img[:,:,0].mean())
            return img[:,:,0]

        elif type in ["train", "test", "validation"]:
            try:
                #  standarsize train values to an interval of [-1,1]
                if img.std() != None :
                    return (img - img.mean()) / img.std()
            except:
                print("NONE IMG")
                return (img - img.mean()) / 1

    # returns an iterator which provides on every call one batch
    # int bsize - batch size to be returned
    # int seed - needed in order to get a unison shuffle
    def getNextBatchTrain(self, bsize, randSeed):
        #batchCount = list(range(int(self.config["trainSize"]*(int(self.config["size"])/bsize))))
        trainElements = list(range(int(self.config["trainSize"]/bsize)))
        #batchCount = self.config["trainSize"]/bsize
        np.random.seed(randSeed)
        np.random.shuffle(trainElements)
        
        for x in trainElements:
            data = [[],[]]
            for b in range(bsize):
                #label
                data[0].append(self.imageData.item().get("trainLabel")[x+b] if self.config["serializedObject"] else self.getImage(x+b, "trainLabel"))
                #train
                data[1].append(self.imageData.item().get("train")[x+b] if self.config["serializedObject"] else self.getImage(x+b, "train"))

            yield np.asarray(data[0]), np.asarray(data[1])

    # returns a batch of test data
    # int testsize - size of the to be returned test data
    def getTestData(self, testsize):
        # test is considered one batch
        data = [[],[]]
        for x in range(testSize):
            #label
            data[0].append(self.imageData.item().get("testLabel")[x] if self.config["serializedObject"] else self.getImage(x, "testLabel"))
            #train
            data[1].append(self.imageData.item().get("test")[x] if self.config["serializedObject"] else self.getImage(x, "test"))
                
        return np.asarray(data[0]), np.asarray(data[1])
    
    # returns a generator of test data
    # int testsize - size of the to be returned test data
    def getNextBatchTest(self, bsize, testsize):
        # test is considered one batch
        for x in range(int(testsize/bsize)):
            data = [[],[]]
            for i in range(bsize):
                #label
                data[0].append(self.imageData.item().get("testLabel")[x+i] if self.config["serializedObject"] else self.getImage(x+i, "testLabel"))
                #train
                data[1].append(self.imageData.item().get("test")[x+i] if self.config["serializedObject"] else self.getImage(x+i, "test"))

            yield np.asarray(data[0]), np.asarray(data[1])

    # returns a batch of validation data
    # int validation size - size of the to be returned validation data
    def getValidationData(self, validationSize):
        data = [[],[]]
        for x in range(validationSize):
            #label
            data[0].append(self.imageData.item().get("validationLabel")[x] if self.config["serializedObject"] else self.getImage(x, "validationLabel"))
            #train
            data[1].append(self.imageData.item().get("validation")[x] if self.config["serializedObject"] else self.getImage(x, "validation"))
            
        return np.asarray(data[0]), np.asarray(data[1])

    # returns a batch of validation data
    # int validation size - size of the to be returned validation data
    def getNextBatchValidation(self, bsize, validationSize):
        for x in range(int(validationSize/bsize)):
            data = [[],[]]
            for i in range(bsize):
                #label
                data[0].append(self.imageData.item().get("validationLabel")[x+i] if self.config["serializedObject"] else self.getImage(x+i, "validationLabel"))
                #train
                data[1].append(self.imageData.item().get("validation")[x+i] if self.config["serializedObject"] else self.getImage(x+i, "validation"))

            yield np.asarray(data[0]), np.asarray(data[1])
