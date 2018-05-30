import tensorflow as tf

STRIDE = 2

def weight_variable(shape, name):
  #initial = tf.truncated_normal(shape, stddev=0.1, name=name)
  #return tf.Variable(initial)
  return tf.get_variable(name, shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())#tf.initializers.orthogonal())

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape, name=name)
  return tf.Variable(initial)

def conv(input, filter, name, pad="SAME", dilation=0, leakyR=False, dropR=0.25):
    f = weight_variable(filter, name+"f1")

    if not dilation > 0:
        conv = tf.nn.conv2d(input, f, strides=[1,1,1,1], padding=pad, name=name)
    else:
        conv = tf.nn.atrous_conv2d(input, f, dilation, padding=pad)
    
    conv_bias = tf.nn.bias_add(conv, bias_variable([filter[3]], name=name+"b1"))
    batch_norm = tf.contrib.layers.batch_norm(conv_bias)
    
    relu = tf.nn.relu(batch_norm)
    if leakyR:
        relu = tf.nn.leaky_relu(batch_norm,alpha=0.2,name=None)

    #drop = tf.nn.dropout(relu, dropR)

    print(name +": ", conv.get_shape())
    #tf.summary.scalar(name, tf.reduce_sum(conv))

    return relu
    
def pool(input, window, stride, poolIndices=False, name="POOL"):
    

    if poolIndices:
        pool = tf.nn.max_pool_with_argmax(
                input,
                ksize=[1, window, window,1],
                strides=[1, stride, stride, 1],
                padding="SAME",
                name=name
            )
    else:
        pool = tf.nn.max_pool(
                input,
                ksize=[1, window, window,1],
                strides=[1, stride, stride, 1],
                padding="SAME",
                name=name
            )

        print(name + ": ", pool.get_shape())

    
    return pool
    
def deconv_filter(shape, name):
    
    # filter = tf.zeros((
    #     shape[0], # height
    #     shape[1], # width
    #     shape[2], # out channels
    #     shape[3] # in channels
    #  ), name=name) 
    
    #return tf.Variable(filter)
    return tf.get_variable(name, shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())

def deconv(layer, outputShape, filterShape, name):
    filter = deconv_filter(filterShape, "deconvF"+layer.name[:len(layer.name)-2])

    deconv = tf.nn.conv2d_transpose(
            layer,
            filter,
            outputShape,
            strides=[1,STRIDE,STRIDE,1],
            padding="SAME",
            name=name
        )
    
    print(name +": ", deconv.get_shape())
    return deconv
