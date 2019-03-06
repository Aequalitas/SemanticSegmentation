# Author: https://github.com/Aequalitas/
# This file contains the training routine
# One epoch is trained and at the end the validation set is evaluated

import tensorflow as tf 
import time 
import numpy as np 
import cv2
from predict import predict
import datetime

def doTrain(epoch, sess, graph, config, data, modelFileName):

    step = 1
    loss = []
    train_acc = []
    acc = []
    epochSize = int(data.config["trainSize"]/config["batchSize"])
    iterator = graph["preFetchIterators"][0]
    nextImgData = iterator.get_next()
  
    for batchIdx in range(epochSize):
        start = time.time()
        try:
            imgData  = sess.run(nextImgData)
            # in case the last rest does not fit into a batch
            if imgData[0].shape[0] == config["batchSize"]:
                _imageData = imgData[0]
                _labelData = imgData[1]
            else:
                break       
        except tf.errors.OutOfRangeError:
            break

        feed_dict = {
            graph["imagePlaceholder"] : _imageData,
            graph["labelPlaceholder"] : _labelData
        }

        # main train operation 
        graph["trainOp"].run(feed_dict=feed_dict)
        end = time.time() 
        
        # print the train status every 10% of the train steps
        if step % int((epochSize)/10) == 0:
            summary, _loss, _train_acc = sess.run([graph["mergedLog"], graph["loss"], graph["accuracy"]], feed_dict=feed_dict)
            train_acc.append(_train_acc*100)
            loss.append(_loss)
            graph["logWriter"].add_summary(summary, step)
            status = "Epoch: "+str(epoch)+" || Step: "+str(step)+"/"+ str(epochSize)
            status += " || loss: "+str(round(np.mean(np.array(loss)), 5))+" || train_acc: "+ str(round(np.mean(np.array(train_acc)), 5))
            status += "% || ETA: "+str(datetime.timedelta(seconds=((end-start)*((epochSize)-step))))
            # ends with \r to delete the older line so the new line can be printed thus only one line is present at a time
            print(status, end="\r")
        
        if step >= epochSize:
            break

        step+=1

    # validate trained model after one epoch
    iterator = graph["preFetchIterators"][1]
    nextImgData = iterator.get_next()
    valSize = int(data.config["validationSize"]/config["batchSize"])
    for r in range(valSize):
        imgData  = sess.run(nextImgData)
        if imgData[0].shape[0] == config["batchSize"]:
            feed_dict={
                graph["imagePlaceholder"]: np.expand_dims(imgData[0], axis=3) if data.config["imageChannels"] == 1 else imgData[0],
                graph["labelPlaceholder"]: imgData if data.config["imageChannels"] == 1 else imgData[1]
            }
            _acc = 100*(graph["accuracy"].eval(feed_dict=feed_dict))    
            acc.append(_acc)

    acc = round(np.mean(np.array(acc)), 5)
    print("\nvalidation_accuracy: "+str(acc))
    return acc
