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
    feed_dict = {
        graph["imagePlaceholder"]: np.expand_dims(imgRes, axis=0)
    }

    predClasses = sess.run(graph["prediction"], feed_dict=feed_dict)
    predClasses = predClasses.reshape(data.config["x"]*data.config["y"])
    predImg = np.zeros((data.config["x"]*data.config["y"],3))


    for idx, p in enumerate(predClasses):
        predImg[idx] = data.config["ClassToRGB"][p]

   
    predImg = predImg.reshape((data.config["y"], data.config["x"], data.config["imageChannels"])).astype("uint8")
    print(predClasses)
    savePath = "../results/"+data.config["name"]+str(data.config["x"])+str(data.config["y"])+config["neuralNetwork"]+".png"
    savedImage = Image.fromarray(predImg, "RGB")
    savedImage.save(savePath)
    display(savedImage)
