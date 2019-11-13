import tensorflow as tf
import nnUtils as util
from nets.mobilenet import mobilenet_v2

STRIDE = 2
K = 3

def net(image, classes):

    
    #encoding - convolution/pooling
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
        logits, endpoints = mobilenet_v2.mobilenet(image, num_classes=None)

    logits = endpoints["layer_10/output"]
    print(logits.get_shape())
    #new_size = (16,32)
    #resize = tf.image.resize(logits, new_size, align_corners=True)
    #conv = util.conv(resize, [3,3,512,320], "up_1", pad="SAME")
    #new_size = (64,128)
    #resize = tf.image.resize(logits, new_size, align_corners=True)
    #conv = util.conv(resize, [3,3,256,512], "up_2", pad="SAME")
          
    new_size = (192,256)
    resize = tf.image.resize(logits, new_size, align_corners=True)
    conv = util.conv(resize, [3,3,128,256], "up_3", pad="SAME")
    
    conv6 = util.conv(conv, [1,1,128,classes], "c6", pad="SAME")

    softmax = tf.nn.softmax(conv6)

    return conv6, tf.argmax(softmax, axis=3), softmax