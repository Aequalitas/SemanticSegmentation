# Author: https://github.com/Aequalitas
# Semantic Segmentation with Tensorflow
# This is the main file which controls which mode of a deep learning process ist used
# MODES:
# train - trains the new initialized or already trained weights of a specific neural network 
# classWeights - creates different class weights for a dataset
# serialize - serializes a dataset into one numpy object file
# eval - evaluates the current neural network on the given test part of the dataset with different metrics
# predict - predicts one default image or with a specified image
# demo - the model predicts all the images in the test part of the dataset. The result masks, images and gif are saved under results/demo/
# profile - calculates the amount of FLOPS for the current neural network
# function deepSS is called like deepSS("train", "netFCN")

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
#from demo import demo

def deepSS(MODE, networkName):

    #load config for tensorflow procedure from json
    config = json.load(open("nets/"+networkName+"Config.json"))
    # load data object initially which provides training and test data loader
    with tf.device('/cpu:0'):
        data = Data("../data/"+config["dataset"]+"/configData"+config["dataset"]+".json")
    
    if MODE == "classWeights":
        data.getClassWeights("quantile")
        
    elif MODE == "serialize":
        print("Serializing dataset to ",data.config["path"]+data.config["fileName"])
        if data.config["fileName"] != "":
            np.save(data.config["path"]+data.config["fileName"], data.getDataset())
            print("Finished serializing")
        else:
            print("You have to set a filename for the serialized file in the config file!")

    else:

        # set Tensorflow to use the GPU according to the config file
        os.environ["CUDA_VISIBLE_DEVICES"]=config["gpu"]
        tf.logging.set_verbosity(tf.logging.INFO)
        # GPU configuration
        tfConfig = tf.ConfigProto()
        # Tensorflow can use 0.6 of the GPU memory
        #tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.6
        # Tensorflow only uses as much GPU memory as is needed
        tfConfig.gpu_options.allow_growth = True

        with tf.Session(config=tfConfig) as sess:
            # create the static tensorflow graph
            graph = buildGraph(sess, data, config)
            sess.run(tf.global_variables_initializer())
            # define the filename which is used to save the model
            modelFileName = "../models/model"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+"Batch"+str(config["batchSize"])+config["neuralNetwork"]+".ckpt"
            try:
                graph["saver"].restore(sess, modelFileName)
            except:
                print("No valid checkpoint found")

            if MODE == "train":
                print("Starting training...")
                best_acc = 0
                LRcounter = 0
                bestMeanIoU = 0
                for e in range(1, config["epochs"]+1):
                    curr_acc = doTrain(e, sess, graph, config, data, modelFileName)
                    predict(sess, config, data, graph)
                    # if validation accuracy is not increasing after 4 times then decrease the learning rate by multiple of 0.1
                    if best_acc < curr_acc:
                        print("val acc of ", curr_acc, " better than ", best_acc)
                        best_acc = curr_acc
                        LRcounter = 0
                        graph["saver"].save(sess, modelFileName)
                    else:
                        print("val acc of ", curr_acc, " NOT better than ", best_acc)
                        if LRcounter >= 4:
                            lr = graph["learningRate"].eval()
                            graph["learningRate"] = tf.assign(graph["learningRate"], lr*0.1)
                            print("Learning rate of ", lr ," is now decreased to ", lr * 0.1)
                            LRcounter = 0

                        LRcounter = LRcounter + 1

                save_path = graph["saver"].save(sess, modelFileName+"End")
                print("Model saved in file: %s" % save_path)

            elif MODE == "eval":
                evaluate(sess, config, data, graph)
           
            elif MODE == "predict":
                predict(sess, config, data, graph)
           
            elif MODE == "demo":
                demo(sess, config, data, graph)
                
            elif MODE == "profile":
                import importlib
                K = tf.keras.backend
                run_meta = tf.RunMetadata()
                K.set_session(sess)
                net = importlib.import_module("nets."+config["neuralNetwork"]).net
                opts = tf.profiler.ProfileOptionBuilder.float_operation()    
                flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
                opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
                params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
                print("{:,} FLOPS --- {:,} total parameters".format(flops.total_float_ops, params.total_parameters))




