import tensorflow as tf
import nnUtils as util

STRIDE = 1

# single cascaded refineNet according to
# Lin et al. - RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
# http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf
def net(image, classes):

    bsize = image.get_shape()[0].value
    f = 3 # kernel size

    #residual convolution unit (RCU)
    #residual block 1
    conv1_residual1 = util.conv(image, [f,f, 3, 128], "conv1_residual1")
    conv2_residual1 = util.conv(conv1_residual1, [f,f, 128, 128], "conv2_residual1")
        
    #residual block 2
    conv1_residual2 = util.conv(conv2_residual1, [f,f, 128, 128], "conv1_residual2")
    conv2_residual2 = util.conv(conv2_residual1, [f,f, 128, 128], "conv2_residual2")
    
    residual1Sum = tf.add(conv2_residual1, conv2_residual2)

    # Multi resolution Fusion - in single cascaded this is just one conv layer, no deconv

    conv1 = util.conv(residual1Sum, [f,f, 128, 256], "conv1")

    # Chained residual pooling

    chainedResidual_p1 = util.pool(conv1, 5, STRIDE, name="convResidual_p1") 
    chainedResidual_conv1 = util.conv(chainedResidual_p1, [f,f, 256, 256], "convResidual_conv1")

    chainedResidualSum1 = tf.add(conv1, chainedResidual_conv1)

    chainedResidual_pool2 = util.pool(chainedResidual_conv1, 5, STRIDE, name="chainedResidual_pool2") 
    chainedResidual_conv1 = util.conv(chainedResidual_pool2, [f,f, 256, 256], "chainedResidual_conv2")

    chainedResidualSum2 = tf.add(chainedResidualSum1, chainedResidual_conv1)

    drop = tf.nn.dropout(chainedResidualSum2, 0.80)

    # Output RCU
    out_conv1_residual1 = util.conv(drop, [f,f, 256, 128], "out_conv1_residual1")
    out_conv2_residual1 = util.conv(out_conv1_residual1, [f,f, 128, classes], "out_conv2_residual1")
    
    softmax = tf.nn.softmax(out_conv2_residual1)

    return out_conv2_residual1, tf.argmax(softmax, axis=3)

