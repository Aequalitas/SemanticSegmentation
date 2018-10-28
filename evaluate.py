import tensorflow as tf 
import sys
import numpy as np 
import csv 
from PIL import Image
from metricsSemSeg import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU

# simple test function, tests all images in testdata once
def evaluate(sess, config, data, graph):

    totalCorrect = 0
    totalCount = data.config["testSize"]*data.config["x"]*data.config["y"]*data.config["imageChannels"]

    totalPAcc = 0.0
    totalMAcc = 0.0
    totalMIU = 0.0
    totalFWIU = 0.0

    i = 0

    for labelData, imgData in data.getNextBatchTest(config["batchSize"], data.config["testSize"]):
        
        # upper = config["batchSize"]*i
        # lower = config["batchSize"]*(i-1)  

        # if upper <= totalTestCount:
        #     images = imgData[lower:upper]
        #     labels = labelData[lower:upper]

        #     totalCount += labels.size
            
            feed_dict = {
                graph["imagePlaceholder"]: imgData
            }

            predClasses = sess.run(graph["prediction"], feed_dict=feed_dict)
            
            predClasses = np.squeeze(predClasses[0])
            labelData = np.squeeze(labelData[0])

            totalCorrect += (predClasses == labelData).sum()

            if i % 50 == 0:
                print("Image ", i, " evaluated...")

            totalPAcc = pixel_accuracy(predClasses, labelData) if totalPAcc == 0.0 else  (totalPAcc + pixel_accuracy(predClasses, labelData))/2
            totalMAcc = mean_accuracy(predClasses, labelData) if totalMAcc == 0.0 else  (totalMAcc + mean_accuracy(predClasses, labelData))/2
            totalMIU = mean_IU(predClasses, labelData) if totalMIU == 0.0 else  (totalMIU + mean_IU(predClasses, labelData))/2
            totalFWIU = frequency_weighted_IU(predClasses, labelData) if totalFWIU == 0.0 else  (totalFWIU + frequency_weighted_IU(predClasses, labelData))/2

            i = i+1

    print("Model Prediction: ", totalCorrect, "/", totalCount, " --> ", (totalCorrect/totalCount)*100,"% pixel correct")
    print("Pixel accuracy: ", totalPAcc ," || Mean accuracy: ", totalMAcc ," || Mean intersection union:", totalMIU ," || frequency weighted IU: ", totalFWIU)