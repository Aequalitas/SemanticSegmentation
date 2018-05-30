import tensorflow as tf 
import sys
import numpy as np 
import csv 
from PIL import Image
from metricsSemSeg import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU

# simple evaluate function, tests all images in testdata once
def evaluate(sess, config, data, graph):

    
    totalTestCount  = int((data.config["size"] - (data.config["size"] * data.config["trainsize"])))
    totalCorrect = 0
    totalCount = 0

    totalPAcc = 0.0
    totalMAcc = 0.0
    totalMIU = 0.0
    totalFWIU = 0.0


    # already standardasized data
    imgData = data.getNextBatchTest(totalTestCount)[1]

    labelData = data.getNextBatchTest(totalTestCount)[0]

    for i in range(1, totalTestCount):
        
        upper = config["batchSize"]*i
        lower = config["batchSize"]*(i-1)  

        if upper <= totalTestCount:
            images = imgData[lower:upper]
            labels = labelData[lower:upper]

            totalCount += labels.size
            
            feed_dict = {
                graph["imagePlaceholder"]: images
            }

            predClasses = sess.run(graph["prediction"], feed_dict=feed_dict)
            
            predClasses = np.squeeze(predClasses)
            labels = np.squeeze(labels)

            totalCorrect += (predClasses == labels).sum()

            if i % 50 == 0:
                print("Image ", i, " tested...")

            totalPAcc = pixel_accuracy(predClasses, labels) if totalPAcc == 0.0 else  (totalPAcc + pixel_accuracy(predClasses, labels))/2
            totalMAcc = mean_accuracy(predClasses, labels) if totalMAcc == 0.0 else  (totalMAcc + mean_accuracy(predClasses, labels))/2
            totalMIU = mean_IU(predClasses, labels) if totalMIU == 0.0 else  (totalMIU + mean_IU(predClasses, labels))/2
            totalFWIU = frequency_weighted_IU(predClasses, labels) if totalFWIU == 0.0 else  (totalFWIU + frequency_weighted_IU(predClasses, labels))/2

    print("Model Prediction: ", totalCorrect, "/", totalCount, " --> ", (totalCorrect/totalCount)*100,"% pixel correct")
    print("Pixel accuracy: ", totalPAcc ," || Mean accuracy: ", totalMAcc ," || Mean intersection union:", totalMIU ," || frequency weighted IU: ", totalFWIU)