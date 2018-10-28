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
    if MODE == 'train' or MODE == 'predict' or MODE == 'eval' or MODE == 'serialize' or MODE == "runLenEncode":
        print("MODE: ", MODE)
    else:
        raise Exception("Provide one argument: train, eval, predict, runLenEncode or serialize!")


    #load config for tensorflow procedure from json
    # networkName = sys.argv[2]
    config = json.load(open("nets/"+networkName+"Config.json"))
    # load data object initially which provides training and test data loader
    data = Data("../data/"+config["dataset"]+"/configData"+config["dataset"]+".json")

    if MODE == "serialize":
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

                for e in range(1, config["epochs"]+1):
                    doTrain(e, sess, graph, config, data, modelFileName)
                    
            elif MODE == "eval":
                evaluate(sess, config, data, graph)
            elif MODE == "predict":
                predict(sess, config, data, graph)
            elif MODE == "runLenEncode":
                runLenEncodeTestSet(sess, config, data, graph)
