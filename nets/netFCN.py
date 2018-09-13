import tensorflow as tf
import nnUtils as util

STRIDE = 2
K = 3

def net(image, classes):

    #encoding - convolution/pooling
    conv1 = util.conv(image, [K,K,3,128], "c1", pad="SAME")
    pool1 = util.pool(conv1, 1, STRIDE, name="p1") 
    
    conv2 = util.conv(pool1, [K,K,128,256], "c2", pad="SAME")
    pool2 = util.pool(conv2, 1, STRIDE, name="p2")  
    
    conv3 = util.conv(pool2, [K,K,256,256], "c3", pad="SAME")
    pool3 = util.pool(conv3, 1, STRIDE, name="p2")
    
    conv4 = util.conv(pool3, [K,K, 256, 512], "c4", pad="SAME")

    #decoding - deconvolution/transposing

    deconv1 = util.deconv(conv4, tf.shape(conv3), [K,K,256,512], "dc1")    
    deconv2 = util.deconv(deconv1, tf.shape(conv2), [K,K,256,256], "dc2")
    deconv3 = util.deconv(deconv2, tf.shape(conv1), [K,K,128,256], "dc3") 
    conv6 = util.conv(deconv3, [1,1,128,classes], "c6", pad="SAME")

    softmax = tf.nn.softmax(conv6)

    return conv6, tf.argmax(softmax, axis=3)