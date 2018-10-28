import tensorflow as tf 
import time 
import numpy as np 
import datetime

from predict import predict


def doTrain(epoch, sess, graph, config, data, modelFileName):

    accuracySum = 0
    step = 1

    loss = []
    train_acc = []
    acc = []

    feed_dict = {}

    for batch in data.getNextBatchTrain(config["batchSize"], step*epoch):
        

        # images that create error which sets the loss to NaN
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
            
            
            #_logits, _loss, _prediction = sess.run([graph["logits"], graph["loss"], graph["prediction"]], feed_dict=feed_dict)
            #print("Image: ",_image.mean(), "Label: ", _labels.mean(),"logits: ", _logits.mean(),"loss: ", _loss,"prediction: ", _prediction.mean())
            
        else:
            print("SKIPPED INVALID STEP: ", step)    
        
        if step % 100 == 0:
            summary, _loss, _train_acc = sess.run([graph["mergedLog"], graph["loss"], graph["accuracy"]], feed_dict=feed_dict)
            
            
            train_acc.append(_train_acc*100)
            loss.append(_loss)
            
            graph["logWriter"].add_summary(summary, step)


            status = "Epoch : "+str(epoch)+" || Step: "+str(step)+"/"+ str(data.config["trainSize"]/config["batchSize"])
            status += " || loss:"+str(round(np.mean(np.array(loss)), 5))+" || train_accuracy:"+ str(round(np.mean(np.array(train_acc)), 5))
            status += "% || ETA: "+str(datetime.timedelta(seconds=((end-start)*((data.config["trainSize"]/config["batchSize"])-step))))
            #status += "% || time 1 step with batch of "+str(config["batchSize"])+": "+str(round(end-start, 3))

            # ends with \r to delete the older line so the new line can be printed
            print(status, end="\r")            
            predict(sess, config, data, graph)
            
        if step % 2000 == 0:
            if _loss > loss[-1]:
                save_path = graph["saver"].save(sess, modelFileName)
                print("\nModel saved in file: %s" % save_path)
            else:
                print("Loss did not advance therefore not saving model")
                
        if step >= data.config["size"]:
            break

        step+=1
    

    # validate trained model after one epoch
    for validationData in data.getNextBatchValidation(config["batchSize"], data.config["validationSize"]):

        feed_dict={
            graph["imagePlaceholder"]: np.expand_dims(validationData[1], axis=3) if data.config["imageChannels"] == 1 else validationData[1],
            graph["labelPlaceholder"]: validationData if data.config["imageChannels"] == 1 else validationData[0]
        }

        _acc = 100*(graph["accuracy"].eval(feed_dict=feed_dict))    
        acc.append(_acc)

    print("\nvalidation_accuracy: "+str(round(np.mean(np.array(acc)), 3)))
    return accuracySum