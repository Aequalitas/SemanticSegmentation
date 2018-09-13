import tensorflow as tf
import nnUtils as util

STRIDE = 2
K = 3
# FC DenseNet according to 
# Jègou et al. - The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
# https://arxiv.org/pdf/1611.09326.pdf

def transitionDown(input, name):
    conv = util.conv(input, [1,1, input.get_shape()[3], input.get_shape()[3]], name+"_c", pad="SAME")
    drop = tf.nn.dropout(conv, 0.2)
    return util.pool(drop, 2, STRIDE, name=name+"p")

def denseBlock(input, layerCount, depth, name, _transitionDown=True):

    currLayer = input

    for x in range(layerCount):
        drop = tf.nn.dropout(currLayer, 0.2)
        currLayer = util.conv(drop,  [K,K, currLayer.get_shape()[3], depth], name+"_c_layer"+str(x)+"_"+str(layerCount), pad="SAME")


    currLayer = transitionDown(currLayer, "TD_"+str(x)+"_"+str(layerCount)) if _transitionDown else currLayer

    return currLayer

def getOutputShape(targetShape, targetDepth):
    return tf.TensorShape([targetShape[0], targetShape[1], targetShape[2], targetDepth])

def net(image, classes):

    initDepth = 128

    conv1 = util.conv(image, [K,K,3,initDepth], "c1", pad="SAME")
    
    # transition downwards
    dB1 = denseBlock(conv1, 4, initDepth*2, "DB1")
    dB2 = denseBlock(dB1, 5, initDepth*3, "DB2")
    dB3 = denseBlock(dB2, 7, initDepth*4, "DB3")
    dB4 = denseBlock(dB3, 10, initDepth*5, "DB4")
    dB5 = denseBlock(dB4, 12, initDepth*6, "DB5")
    dB6 = denseBlock(dB5, 15, initDepth*7, "DB6", _transitionDown=False)

    # transition upwards 
    deconv1 = util.deconv(dB6, getOutputShape(dB4.get_shape(),initDepth*6), [K,K,initDepth*6,initDepth*7], "dc1")    
    dB7 = denseBlock(deconv1, 12, initDepth*6, "DB7", _transitionDown=False)    
    deconv2 = util.deconv(dB7, getOutputShape(dB3.get_shape(),initDepth*5), [K,K,initDepth*5,initDepth*6], "dc2")    
    dB8 = denseBlock(deconv2, 10, initDepth*5, "DB8", _transitionDown=False)
    deconv3 = util.deconv(dB8, getOutputShape(dB2.get_shape(),initDepth*4), [K,K,initDepth*4,initDepth*5], "dc3")    
    dB9 = denseBlock(deconv3, 7, initDepth*4, "DB9", _transitionDown=False)
    deconv4 = util.deconv(dB9, getOutputShape(dB1.get_shape(),initDepth*3), [K,K,initDepth*3,initDepth*4], "dc4")    
    dB10 = denseBlock(deconv4, 5, initDepth*3, "DB10", _transitionDown=False)
    deconv5 = util.deconv(dB10, getOutputShape(conv1.get_shape(), initDepth*2), [K,K,initDepth*2,initDepth*3], "dc5")    

    # no denseBlock function for the last denseBlock because of structure within nnUtils
    # these convs contain bNorm and relu after conv
    conv2 = tf.nn.conv2d(deconv5, util.weight_variable([K,K,initDepth*2, initDepth], "c2"), strides=[1,1,1,1], padding="SAME")
    drop = tf.nn.dropout(conv2, 0.2)

    conv3 = tf.nn.conv2d(drop, util.weight_variable([1,1, initDepth, classes], "c3"), strides=[1,1,1,1], padding="SAME")

    softmax = tf.nn.softmax(conv3)

    return conv3, tf.argmax(softmax, axis=3), softmax