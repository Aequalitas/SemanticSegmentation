import tensorflow as tf 
import sys
import numpy as np 
import csv
import cv2
from PIL import Image
from IPython.display import display 

def predict(sess, config, data, graph):
    imagePath = "../results/predict"+config["dataset"]+".jpg"

    img = cv2.imread(imagePath)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRes = cv2.resize(img, (data.config["x"], data.config["y"]), interpolation=cv2.INTER_NEAREST)
    
    imgRes = (imgRes - imgRes.mean()) / imgRes.std()
    
    inputData = np.expand_dims(imgRes, axis=0)
    
    if config["batchSize"] > 1:
        fillerArr = np.zeros((1,data.config["y"], data.config["x"], data.config["imageChannels"]))
        for x in range(config["batchSize"]-1):
            inputData = np.concatenate((inputData, fillerArr), axis=0)
  
    feed_dict = {
            graph["imagePlaceholder"]: inputData 
        }
                       
    predClasses = sess.run(graph["prediction"], feed_dict=feed_dict)
    predClasses = predClasses[0].reshape(data.config["x"]*data.config["y"])
    predImg = np.zeros((data.config["x"]*data.config["y"],3))


    #for idx, p in enumerate(predClasses):
    #    predImg[idx] = data.config["ClassToRGB"][p]

    for cl in range(config["classes"]):
        predImg[(predClasses == cl)] = data.config["ClassToRGB"][cl]
   
    predImg = predImg.reshape((data.config["y"], data.config["x"], data.config["imageChannels"])).astype("uint8")
    #print(predClasses)
    savePath = "../results/"+data.config["name"]+str(data.config["x"])+str(data.config["y"])+config["neuralNetwork"]+".png"
    savedImage = Image.fromarray(predImg, "RGB")
    savedImage.save(savePath)
    #display(savedImage)
