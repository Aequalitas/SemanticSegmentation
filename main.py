# Semantic Segmentation with Tensorflow
# Franz Weidmann www.github.com/Aequalitas/SemanticSegmentation

import tensorflow as tf
import os
import numpy as np
import sys
from data import Data
import json
from graph import buildGraph
from train import doTrain
from predict import predict
from evaluate import evaluate

MODE = sys.argv[1]
if MODE == 'train' or MODE == 'predict' or MODE == 'eval' or MODE == 'serialize':
    print("MODE: ", MODE)
else: 
    raise Exception("Provide one argument: train, eval, predict or serialize!")

# load config for tensorflow procedure from json
config = json.load(open(sys.argv[2]+"Config.json"))
# load data object initially which provides training and test data loader
data = Data("configData.json", sys.argv[3])

if MODE == "serialize":
    if data.config["fileName"] != "":
        np.save(data.config["fileName"], data.getDataset())
    else:
        print("You have to set a filename for the serialized file in the config file!")
    exit()

# create the tensorflow graph and logging
graph = buildGraph(data, config)


os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[4]#config["gpu"]
tf.logging.set_verbosity(tf.logging.INFO)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    modelFileName = "../../models/model"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+config["neuralNetwork"]+".ckpt"
    try:
        graph["saver"].restore(sess, modelFileName)
    except:
        print("No valid checkpoint found")

    accuracySum = 0

    if MODE == "train":
        print("Starting training...")

        for e in range(1, config["epochs"]+1):
            accuracySum += doTrain(e, sess, graph, config, data, modelFileName)
            #print("\nMean Accuracy for ",e," epochs: ", (accuracySum/(e)))
            save_path = graph["saver"].save(sess, modelFileName)
            print("\nModel saved in file: %s" % save_path)

        save_path = graph["saver"].save(sess, modelFileName)
        print("Model saved in file: %s" % save_path)
     
    elif MODE == "eval":
        evaluate(sess, config, data, graph)
    elif MODE == "predict":
        predict(sess, config, data, graph)