import tensorflow as tf
import nnUtils as util

STRIDE = 1
# kernel size of the conv filter
K = 3
# baseDepth
initDepth = 256
# Multi-Path refinement
# Lin et al. - RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
# http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf
# difference to standard version:
#  - with two inputs at every pooling step
#  - halved number of features
#  - stage 2 and 1 are added to gain original resolution and further details

def residualConvUnit(pooledImage, downsampleLevel, number):

    relu1 = tf.nn.relu(pooledImage)
    name = "conv1_residual1_"+downsampleLevel+"_"+number
    f = util.weight_variable([K,K, pooledImage.get_shape()[3], initDepth], name+"_f1")
    conv1_residual1 = tf.nn.conv2d(relu1, f, strides=[1,1,1,1], padding="SAME", name=name)

    relu2 = tf.nn.relu(conv1_residual1)
    name = "conv1_residual2_"+downsampleLevel+"_"+number
    f = util.weight_variable([K,K,initDepth,initDepth], name+"_f2")
    return tf.nn.conv2d(relu2, f, strides=[1,1,1,1], padding="SAME", name=name)

def adaptiveConv(input, downsampleLevel, downsample=True):
    
    if downsample:
        #downsample image
        input = util.pool(input, 1, int(downsampleLevel), name="RCUPool_"+downsampleLevel)

    #residual convolution unit (RCU) 1
    firstRCU = residualConvUnit(input, downsampleLevel, "1")
    #residual convolution unit (RCU) 2
    secondRCU = residualConvUnit(firstRCU, downsampleLevel, "2")
    
    return tf.add(firstRCU, secondRCU)

def multiResolutionFusion(inputs, downsampleLevel):

    name = "conv1_"+downsampleLevel
    f = util.weight_variable([K,K,initDepth,initDepth*2], name+"_f1")
    conv1 = tf.nn.conv2d(inputs[0], f, strides=[1,1,1,1], padding="SAME", name=name)
    
    if inputs[1] == None:
        return conv1
    else:
        name = "conv2_"+downsampleLevel
        f = util.weight_variable([K,K,initDepth,initDepth], name+"_f2")
        conv2 = tf.nn.conv2d(inputs[1], f, strides=[1,1,1,1], padding="SAME", name=name)
        outputShape = [inputs[0].get_shape()[0].value, inputs[0].get_shape()[1].value, inputs[0].get_shape()[2].value, initDepth*2]
        deconv1 = util.deconv(conv2, outputShape, [K,K,initDepth*2,initDepth], "dc1_"+downsampleLevel, stride=2) 
        return tf.add(conv1, deconv1)

def refineNet(inputs, downsampleLevel):

    bsize = inputs[0].get_shape()[0].value

    currLevelFeatures = adaptiveConv(inputs[0], downsampleLevel)
    if not downsampleLevel == "32":
        previousFeatures = adaptiveConv(inputs[1], "prev"+downsampleLevel, downsample=False)
    else:
        previousFeatures = None

    # Multi resolution Fusion with current resolution and the previous one
    mRF = multiResolutionFusion([currLevelFeatures, previousFeatures], downsampleLevel)
    
    # Chained residual pooling
    reluRCP = tf.nn.relu(mRF)
    chainedResidual_p1 = util.pool(reluRCP, 5, STRIDE, name="convResidual_p1_"+downsampleLevel) 
    chainedResidual_conv1 = util.conv(chainedResidual_p1, [K,K, initDepth*2, initDepth*2], "convResidual_conv1_"+downsampleLevel)

    chainedResidualSum1 = tf.add(reluRCP, chainedResidual_conv1)

    chainedResidual_pool2 = util.pool(chainedResidual_conv1, 5, STRIDE, name="chainedResidual_pool2_"+downsampleLevel) 
    chainedResidual_conv1 = util.conv(chainedResidual_pool2, [K,K, initDepth*2, initDepth*2], "chainedResidual_conv2_"+downsampleLevel)

    chainedResidualSum2 = tf.add(chainedResidualSum1, chainedResidual_conv1)

    drop = tf.nn.dropout(chainedResidualSum2, 0.80)

    # Output RCU
    out_conv1_residual1 = util.conv(chainedResidualSum2, [K,K, initDepth*2, initDepth], "out_conv1_residual1_"+downsampleLevel)
    out_conv2_residual1 = util.conv(out_conv1_residual1, [K,K, initDepth, initDepth], "out_conv2_residual1_"+downsampleLevel)
    
    return out_conv2_residual1


def net(image, classes):

    output_32 = refineNet([image], "32")
    output_16 = refineNet([image, output_32], "16")
    output_8 = refineNet([image, output_16], "8")
    output_4 = refineNet([image, output_8], "4")
    output_2 = refineNet([image, output_4], "2")
    output_1 = refineNet([image, output_2], "1")

    # upscaling to original size for usage
    
    conv1 = adaptiveConv(output_1, "0", downsample=False)

    # outputShape = [image.get_shape()[0].value, image.get_shape()[1].value, image.get_shape()[2].value, initDepth]
    # deconvFinal = util.deconv(conv1, outputShape, [K,K, initDepth,initDepth], "deconvFinal", stride=2)
    convLast = util.conv(conv1, [K,K,initDepth,classes], "lastConv")

    out = tf.nn.softmax(convLast)
    
    return convLast, tf.argmax(out, axis=3), out

