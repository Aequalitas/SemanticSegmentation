import tensorflow as tf 
import time 
import numpy as np 


def doTrain(epoch, sess, graph, config, data, modelFileName):

    step = 1

    loss = None
    acc = None

    for batch in data.getNextBatchTrain(config["batchSize"]):

        # this list contains corrupted images that create error which sets the loss to NaN
        faultyVals = list()

        if step not in faultyVals:
            start = time.time()
            _image = np.expand_dims(batch[1], axis=3) if data.config["imageChannels"] == 1 else batch[1]
            _labels =  batch if data.config["imageChannels"] == 1 else batch[0]

            feed_dict = {
                graph["imagePlaceholder"] : _image,
                graph["labelPlaceholder"] : _labels
            }
            
            graph["trainOp"].run(feed_dict=feed_dict)
            
            end = time.time() 

        else:
            print("SKIPPED INVALID STEP: ", step)    
        
        if step % 100 == 0:
            summary, _loss, train_acc = sess.run([graph["mergedLog"], graph["loss"], graph["accuracy"]], feed_dict=feed_dict)
            
            if loss == None:
                loss = _loss
            else:
                loss = (loss+_loss) / 2
            
            graph["logWriter"].add_summary(summary, step)

            train_acc *= 100

            status = "Epoch : "+str(epoch)+" || Step: "+str(step+1)+"/"+str(int(int(data.config["size"]) * data.config["trainsize"]))
            status += " || loss:"+str(round(loss, 3))+" || train_accuracy:"+ str(round(train_acc, 3))
            status += "% || time 1 step with batch of "+str(config["batchSize"])+": "+str(round(end-start, 3))

            print(status, end="\r")            

        if step >= config["steps"]:
            break

        step+=1

    # validate trained model after epoch
    testData = data.getNextBatchTest(1)#data.getNextBatchTest(testBatch)
    feed_dict={
        graph["imagePlaceholder"]: np.expand_dims(testData[1], axis=3) if data.config["imageChannels"] == 1 else testData[1],
        graph["labelPlaceholder"]: testData if data.config["imageChannels"] == 1 else testData[0]
    }
    
    if acc == None:
        acc = 100*(graph["accuracy"].eval(feed_dict=feed_dict))
    else:
        acc = (100*(graph["accuracy"].eval(feed_dict=feed_dict))+acc) / 2
    
    print("\ntest_accuracy: "+str(round(acc, 3)))