# Semantic Segmentation with Tensorflow
# by Franz Weidmann www.github.com/Aequalitas

import tensorflow as tf
import os
import numpy as np
import sys
import json


from data import Data
from graph import buildGraph
from train import doTrain
from predict import predict
from evaluate import evaluate
from runLengthEncoding import runLenEncodeTestSet


def deepSS(MODE, networkName, GPUNr="0"):

    #MODE = sys.argv[1]
    if MODE == 'train' or MODE == 'predict' or MODE == 'eval' or MODE == 'serialize' or MODE == "runLenEncode" or MODE == "classWeights":
        print("MODE: ", MODE)
    else:
        raise Exception("Provide one argument: train, eval, predict, runLenEncode, classWeights or serialize!")


    #load config for tensorflow procedure from json
    # networkName = sys.argv[2]
    config = json.load(open("nets/"+networkName+"Config.json"))
    # load data object initially which provides training and test data loader
    data = Data("../data/"+config["dataset"]+"/configData"+config["dataset"]+".json")
    
    if MODE == "classWeights":
        data.getClassWeights("1x")
    elif MODE == "serialize":
        print("Serializing dataset to ",data.config["path"]+data.config["fileName"])

        if data.config["fileName"] != "":
            np.save(data.config["path"]+data.config["fileName"], data.getDataset(flipH=False))
            print("Finished serializing!")
        else:
            print("You have to set a filename for the serialized file in the config file!")
        
    else:
        # create the tensorflow graph and logging
        graph = buildGraph(data, config)


        os.environ["CUDA_VISIBLE_DEVICES"]=GPUNr
        tf.logging.set_verbosity(tf.logging.INFO)


        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            modelFileName = "../models/model"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+config["neuralNetwork"]+".ckpt"
            
            try:
                graph["saver"].restore(sess, modelFileName)
            except:
                print("No valid checkpoint found")

            accuracySum = 0

            if MODE == "train":
                 print("Starting training...")

                best_acc = 0
                LRcounter = 0
                for e in range(1, config["epochs"]+1):
                    curr_acc = doTrain(e, sess, graph, config, data, modelFileName)
                    predict(sess, config, data, graph)
                    
                    if best_acc < curr_acc:
                        graph["saver"].save(sess, modelFileName)
                        print("val acc of ", curr_acc, " better than ", best_acc)
                        best_acc = curr_acc
                        LRcounter = 0
                    else:
                        print("val acc of ", curr_acc, " NOT better than ", best_acc)
                        if LRcounter >= 5:
                            lr = graph["learningRate"].eval()
                            graph["learningRate"] = tf.assign(graph["learningRate"], lr*0.1)
                            print("Learning rate of ", lr ," is now decreased to ", lr * 0.1)
                            LRcounter = 0
                        
                        LRcounter = LRcounter + 1
                    
                    if e % 5 == 0:
                        evaluate(sess, config, data, graph)
                
            elif MODE == "eval":
                evaluate(sess, config, data, graph)
            elif MODE == "predict":
                predict(sess, config, data, graph)
            elif MODE == "runLenEncode":
                runLenEncodeTestSet(sess, config, data, graph)
