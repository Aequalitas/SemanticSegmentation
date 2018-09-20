import tensorflow as tf 
import importlib
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
    # onehot_labels = tf.one_hot(labels, CLASSES)
    # weights = onehot_labels * classWeights 
    # weights = tf.reduce_sum(weights, 3)


    # Neural Network
    logits, predictionNet, softmaxNet = net(image, data.config["classes"])

    # Training
    # sparse because one pixel == one class and not multiple
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))#, weights=weights)
    tf.summary.scalar("loss", loss)

    # optimizer
    LR = tf.train.exponential_decay(LR, global_step, 10000, 0.96, staircase=True)
    tf.summary.scalar("learning_rate", LR)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR, name="AdamOpt")
    train_op = optimizer.minimize(loss, global_step=global_step)
    # grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    # train_op = optimizer.apply_gradients(grads)

    correct_prediction = tf.equal(tf.cast(predictionNet, tf.int32), labels)
    # frequency weighted accuracy
    accuracy = tf.reduce_mean(tf.multiply(tf.cast(correct_prediction, tf.float32),1))

    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("../../logs/", graph=tf.get_default_graph())

    return {
        "logits":logits,
        "loss": loss,
        "mergedLog": merged,
        "prediction": predictionNet,
        "softmaxOut": softmaxNet,
        "imagePlaceholder": image,
        "labelPlaceholder": labels,
        "trainOp": train_op,
        "saver": saver,
        "logWriter": writer,
        "accuracy": accuracy
    }