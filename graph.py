import tensorflow as tf 
import importlib
import numpy as np
#from netFCN import net

def buildGraph(data, config):

    LR = config["learningRate"]
    net = importlib.import_module("nets."+config["neuralNetwork"]).net

    # REAL TENSORFLOW - low API

    # Main Variables
    # global step for decaying learning rate
    global_step = tf.Variable(0, trainable=False)

    # create placeholder later to be filled
    imageShape = [config["batchSize"], data.config["y"], data.config["x"], data.config["imageChannels"]]
    image = tf.placeholder(tf.float32, shape=imageShape, name="input_image")

    # has to be reshaped in case output resolution is smaller as in the unet
    labelsShape = [config["batchSize"], data.config["y"], data.config["x"]]
    labels = tf.placeholder(tf.int32, labelsShape, name="labels")

    # class Weights for class imbalance
    # # create weights for this particular training image
    classWeights = np.load("classWeights"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+".npy")
    onehot_labels = tf.one_hot(labels, data.config["classes"])
    weights = onehot_labels * (np.ones((data.config["classes"])) if classWeights is None else classWeights)
    weights = tf.reduce_sum(weights, 3)


    # Neural Network
    logits, predictionNet, softmaxNet = net(image, data.config["classes"])

    # Training
    # sparse because one pixel == one class and not multiple
    #loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights))
    
    tf.summary.scalar("loss", loss)

    # optimizer
    LR = tf.train.exponential_decay(LR, global_step, 2000, 0.96, staircase=True)
    tf.summary.scalar("learning_rate", LR)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR, name="AdamOpt")
    train_op = optimizer.minimize(loss, global_step=global_step)
    # grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    
    # train_op = optimizer.apply_gradients(grads)

    correct_prediction = tf.equal(tf.cast(predictionNet, tf.int32), labels)
    # frequency weighted accuracy
    #accuracy = tf.reduce_mean(tf.multiply(tf.cast(correct_prediction, tf.float32), weights))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #tf.metrics.accuracy(labels, predictionNet, weights)
    
    tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("../logs/", graph=tf.get_default_graph())
    
    labelData = None
    imgData = None
    itInit = None
    if data.config["tfPrefetch"]:
        with tf.device('/cpu:0'):
            # tensorflow dataset for a more efficient input pipeline through threading
            imageFilenames = tf.constant(data.imageData["train"])
            labelsFileNames = tf.constant(data.imageData["trainLabel"])

            dataset = tf.data.Dataset.from_tensor_slices((imageFilenames, labelsFileNames))
            dataset = dataset.map(lambda filename, label: tf.py_func(
                                          data.getImageTuple,
                                          [filename, label],
                                          [tf.float32, tf.int32]
                                       ),  num_parallel_calls=config["threadCount"])


            dataset = dataset.shuffle(buffer_size=int(1000/config["batchSize"]))
            dataset = dataset.batch(config["batchSize"], drop_remainder=True)
            dataset = dataset.prefetch(1)
            iterator = dataset.make_initializable_iterator()
            imgData, labelData = iterator.get_next()
            itInit = iterator.initializer

    
    return {
        "logits":logits,
        "loss": loss,
        "mergedLog": merged,
        "prediction": predictionNet,
        "softmaxOut": softmaxNet,
        "imagePlaceholder": image,
        "labelPlaceholder": labels,
        "preFetchImageData":[imgData, labelData],
        "itInit":itInit,
        "trainOp": train_op,
        "saver": saver,
        "logWriter": writer,
        "accuracy": accuracy
    }