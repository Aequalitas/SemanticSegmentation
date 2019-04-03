# Author: https://github.com/Aequalitas/
# This file builds the static graph of Tensorflow. Means the Neural Network is built,
# the loss function, Optimizer, logging functions and the dataset pipeline with tf.Data

import tensorflow as tf 
import importlib
import numpy as np

def buildGraph(sess, data, config):

        # Main Variables
       
        # create placeholder later to be filled
        imageShape = [config["batchSize"], data.config["y"], data.config["x"], data.config["imageChannels"]]
        image = tf.placeholder(tf.float32, shape=imageShape, name="input_image")

        labelsShape = [config["batchSize"], data.config["y"], data.config["x"]]
        labels = tf.placeholder(tf.int32, labelsShape, name="labels")

        # class Weights for class imbalance
        # create weights for the particular batch
        classWeights = [1.0, 1.0]
        #try:
        #    classWeights = np.load("classWeights"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+".npy")
        #    print("Classweights successfully loaded!")
        #except:
        #       print("No class weights to load!")
        onehot_labels = tf.one_hot(labels, data.config["classes"])
        weights = onehot_labels * classWeights 
        weights = tf.reduce_sum(weights, 3)

        # Neural Network is loaded from an extra file whose name is specified in the config file
        net = importlib.import_module("nets."+config["neuralNetwork"]).net
        logits, predictionNet, softmaxNet = net(image, data.config["classes"])

        # Training part
        # sparse because labels are given as in only the correct class has the value 1 and the rest are zeros
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits,weights=weights))
        #loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
        tf.summary.scalar("loss", loss)

        # Set a learn rate variable for later configuration
        LR = tf.Variable(config["learningRate"], name="learningRate")
        tf.summary.scalar("learning_rate", LR)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=LR, name="AdamOpt")
        train_op = optimizer.minimize(loss, global_step=tf.Variable(0, trainable=False))
        
        # metric variables for train pixel accuracy
        correct_prediction = tf.equal(tf.cast(predictionNet, tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        # Tensorflow model saver
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

        # logger 
        writer = tf.summary.FileWriter("../logs/", graph=tf.get_default_graph())

        # Tensorflow dataset for a more efficient input pipeline by using threads
        labelData = None
        imgData = None
        with tf.device('/cpu:0'):
            iterators = []
            for _type in ["train", "validation", "test"]:

                print("Creating ", _type, " dataset...")
                imageFilenames = tf.constant(data.imageData[_type])
                labelsFileNames = tf.constant(data.imageData[_type+"Label"])

                dataset = tf.data.Dataset.from_tensor_slices((imageFilenames, labelsFileNames))
                dataset = dataset.map(lambda filename, label: tf.py_func(
                                              data.getImageTuple,
                                              [filename, label],
                                              [tf.float32, tf.int32]
                                           ),  num_parallel_calls=config["threadCount"])

                if (_type == "train") | (_type == "validation"):
                    # data augmentation
                    datasetFlippedV = dataset.map(lambda trainImage, labelImage:
                                                 (tf.reverse(trainImage, axis=[1]), tf.reverse(labelImage, axis=[1]))
                                               , num_parallel_calls=config["threadCount"])
                    dataset = dataset.concatenate(datasetFlippedV)

                    #datasetFlippedH = dataset.map(lambda trainImage, labelImage:
                    #                              tf.reverse(trainImage, axis=2), tf.reverse(labelImage, axis=2)
                    #                           , num_parallel_calls=config["threadCount"])

                    dataset = dataset.concatenate(datasetFlippedV)
                    data.config[_type+"Size"] *= 2
                    print("Dataset flipped vertically new ", _type, "Size: ", data.config[_type+"Size"])

                if _type == "train":
                    dataset = dataset.shuffle(buffer_size=int(100/config["batchSize"]))
                
                dataset = dataset.batch(config["batchSize"])
                dataset = dataset.prefetch(4)
                dataset = dataset.repeat(config["epochs"])
                iterators.append(dataset.make_one_shot_iterator())


        return {
            "logits":logits,
            "loss": loss,
            "mergedLog": merged,
            "prediction": predictionNet,
            "softmaxOut": softmaxNet,
            "learningRate": LR,
            "imagePlaceholder": image,
            "labelPlaceholder": labels,
            "trainOp": train_op,
            "preFetchIterators": iterators,
            "saver": saver,
            "logWriter": writer,
            "accuracy": accuracy
        }

