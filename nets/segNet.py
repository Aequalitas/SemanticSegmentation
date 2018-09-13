import tensorflow as tf
import nnUtils as util

# unpooling by Fabian Bormann https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/blob/master/tests/UnpoolLayerTest.ipynb
def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape):
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat([t2, t1], 3)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

STRIDE = 1
# segNet according to:
#  Badrinarayanan et al. - SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7803544
def net(image, classes):

    initDepth = 512
    bsize = image.get_shape()[0].value
    f = 7 # kernel size

    #encoding - downsampling
    # encoding level 1
    e1_c1 = util.conv(image, [f,f,3,initDepth], "e1_c1")
    e1_c2 = util.conv(e1_c1, [f,f,initDepth,initDepth], "e1_c2")
    pool1, pool1_indices = util.pool(e1_c2, 2, 2, poolIndices=True, name="pool1")

    # encoding level 2
    e2_c1 = util.conv(pool1, [f,f,initDepth,initDepth], "e2_c1")
    e2_c2 = util.conv(e2_c1, [f,f,initDepth,initDepth], "e2_c2")
    pool2, pool2_indices = util.pool(e2_c2, 2, 2, poolIndices=True, name="pool2")
    
    # encoding level 3
    e3_c1 = util.conv(pool2, [f,f,initDepth,initDepth], "e3_c1")
    e3_c2 = util.conv(e3_c1, [f,f,initDepth,initDepth], "e3_c2")
    e3_c3 = util.conv(e3_c2, [f,f,initDepth,initDepth], "e3_c3")
    pool3, pool3_indices = util.pool(e3_c3, 2, 2, poolIndices=True, name="pool3")

    # encoding level 4
    e4_c1 = util.conv(pool3, [f,f,initDepth,initDepth], "e4_c1")
    e4_c2 = util.conv(e4_c1, [f,f,initDepth,initDepth], "e4_c2")
    e4_c3 = util.conv(e4_c2, [f,f,initDepth,initDepth], "e4_c3")
    pool4, pool4_indices = util.pool(e4_c3, 2, 2, poolIndices=True, name="pool4")

    # encoding level 5
    e5_c1 = util.conv(pool4, [f,f,initDepth,initDepth], "e5_c1")
    e5_c2 = util.conv(e5_c1, [f,f,initDepth,initDepth], "e5_c2")
    e5_c3 = util.conv(e5_c2, [f,f,initDepth,initDepth], "e5_c3")
    pool5, pool5_indices = util.pool(e5_c3, 2, 2, poolIndices=True, name="pool5")


    #decoding with pool indices
    # decoding level 1
    # upsample with pooling indices
    upSam5 = unpool_layer2x2(pool5, pool5_indices, pool4.get_shape())
    de1_c1 = util.conv(upSam5, [f,f,initDepth,initDepth], "de1_c1")
    de1_c2 = util.conv(de1_c1, [f,f,initDepth,initDepth], "de1_c2")
    
    # dencoding level 2
    upSam4 = unpool_layer2x2(pool4, pool4_indices, pool3.get_shape())
    de2_c1 = util.conv(upSam4, [f,f,initDepth,initDepth], "de2_c1")
    de2_c2 = util.conv(de2_c1, [f,f,initDepth,initDepth], "de2_c2")
    
    # decoding level 3
    upSam3 = unpool_layer2x2(pool3, pool3_indices, pool2.get_shape())
    de3_c1 = util.conv(upSam3, [f,f,initDepth,initDepth], "de3_c1")
    de3_c2 = util.conv(de3_c1, [f,f,initDepth,initDepth], "de3_c2")
    de3_c2 = util.conv(de3_c2, [f,f,initDepth,initDepth], "de3_c3")
    
    # decoding level 4
    upSam2 = unpool_layer2x2(pool2, pool2_indices, pool1.get_shape())
    de4_c1 = util.conv(upSam2, [f,f,initDepth,initDepth], "de4_c1")
    de4_c2 = util.conv(de4_c1, [f,f,initDepth,initDepth], "de4_c2")
    de4_c3 = util.conv(de4_c2, [f,f,initDepth,initDepth], "de4_c3")
    
    # decoding level 5
    upSam1 = unpool_layer2x2(pool1, pool1_indices, e1_c1.get_shape())
    de5_c1 = util.conv(upSam1, [f,f,initDepth,initDepth], "de5_c1")
    de5_c2 = util.conv(de5_c1, [f,f,initDepth,initDepth], "de5_c2")
    de5_c3 = util.conv(de5_c2, [f,f,initDepth,classes], "de5_c3")
    
    softmax = tf.nn.softmax(de5_c3)

    return de5_c3, tf.argmax(softmax, axis=3), softmax

