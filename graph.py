import tensorflow as tf 
import importlib


# builds the tensorflow graph with given config
def buildGraph(data, config):

    LR = config["learningRate"]
    net = importlib.import_module(config["neuralNetwork"]).net

    # Main Variables
    # global step for decaying learning rate
    global_step = tf.Variable(0, trainable=False)

    # create placeholder later to be filled
    imageShape = [config["batchSize"], data.config["y"], data.config["x"], data.config["imageChannels"]]
    image = tf.placeholder(tf.float32, shape=imageShape, name="input_image")

    labelsShape = [config["batchSize"], data.config["y"], data.config["x"]]
    labels = tf.placeholder(tf.int32, labelsShape, name="labels")

    # class Weights for class imbalance
    #classWeights = tf.ones([BSIZE, CLASSES]) * (np.load("classWeights.npy").astype(np.float32))

    # Neural Network
    logits, predictionNet = net(image, data.config["classes"])

    # Training
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)#, weights=weights)
    tf.summary.scalar("loss", loss)

    # optimizer
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
    LR = tf.train.exponential_decay(LR, global_step, 100000, 0.96)
    tf.summary.scalar("learning_rate", LR)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR, name="AdamOpt")
    train_op = optimizer.minimize(loss, global_step=global_step)

    print("Starting training...")

    correct_prediction = tf.equal(tf.cast(predictionNet, tf.int32), labels)
    accuracy = tf.reduce_mean(tf.multiply(tf.cast(correct_prediction, tf.float32),1))

    tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("../../logs/", graph=tf.get_default_graph())

    return {
        "loss": loss,
        "mergedLog": merged,
        "prediction": predictionNet,
        "imagePlaceholder": image,
        "labelPlaceholder": labels,
        "trainOp": train_op,
        "saver": saver,
        "logWriter": writer,
        "accuracy": accuracy
    }