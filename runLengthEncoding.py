# encode an segmented image in to the run length encoding
# where 1 10 would be all further 10 pixel starting at pixel 1 are class 1


import warnings
warnings.filterwarnings('ignore')

import cv2 
import numpy as np 
import csv 
import tensorflow as tf
import os
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

def runLenEncodeTestSet(sess, config, data, graph):
    
    x_valid = []
    y_valid = []
    x_preds = []
    y_preds = []
    
    print("loading data...")

    iterator = graph["preFetchIterators"][2]
    testSize = int(data.config["testSize"]/config["batchSize"])
    for r in range(testSize):

        imgData = iterator.get_next()
        imgData  = sess.run(imgData)

        if imgData[0].shape[0] == config["batchSize"]:
            feed_dict = {
                graph["imagePlaceholder"]: imgData[0]
            }

            pred = graph["softmaxOut"].eval(feed_dict=feed_dict)
            pred = np.argmax(pred, axis=3)
            labels = imgData[1]
     
            for b in range(config["batchSize"]):
                x_preds.append(pred[b].squeeze())
                x_valid.append(imgData[0][b].squeeze())
                y_valid.append(imgData[1][b].squeeze())
                
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_preds = np.array(x_preds)
    y_preds = np.array(y_preds)

    #print(x_preds)
    #print(x_preds.shape)
    #thresholdMask = (x_preds > 0.663)
    #print("Thresholdmask: ", thresholdMask.shape)
    #sanityCheck(x_valid[thresholdMask], y_valid[thresholdMask], x_preds[thresholdMask])
    #sanityCheck(x_valid, y_valid, x_preds)

    thresholdOpt(x_preds, y_valid)


def runLenEncode(segImage):

    segImage = segImage.reshape((data.config["x"]*data.config["y"]))
    runLengthCode = ""
    startPixel = 0
    pixelSteps = 0

    for pIdx, p in enumerate(segImage):
        
        if p == 1:
            if pixelSteps == 0:
                startPixel = pIdx+1
                pixelSteps += 1
            else:
                pixelSteps += 1 
        else:
            if pixelSteps != 0:
                runLengthCode += " "+str(startPixel)
                runLengthCode += " "+str(pixelSteps)
                startPixel = 0
                pixelSteps = 0
    
    if runLengthCode == "":
        print("EMPTY LINE - No salt found")
        return " 1 1"
    else:
        return runLengthCode

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def sanityCheck(x_valid, y_valid, preds_valid):
    print("Sanity Check")
    # display ground-truth
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for idx, i in enumerate(x_valid[:max_images]):
        img = (x_valid[idx] * 255).astype(np.uint8)
        mask = (y_valid[idx]  * 255).astype(np.uint8)
        ax = axs[int(idx / grid_width), idx % grid_width]
        #ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.6, cmap="Greens")
        #ax.imshow(pred, alpha=0.6, cmap="OrRd")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
  
    plt.suptitle("Green: salt")
    plt.show()
    
    #display predictions
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for idx, i in enumerate(x_valid[:max_images]):
        img = (x_valid[idx] * 255).astype(np.uint8)
        pred = (preds_valid[idx] * 255).astype(np.uint8)
        ax = axs[int(idx / grid_width), idx % grid_width]
        #ax.imshow(img, cmap="Greys")
        #ax.imshow(mask, alpha=0.6, cmap="Greens")
        ax.imshow(pred, alpha=0.6, cmap="OrRd")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        
    plt.suptitle("Red: prediction")
    plt.show()

# threshold optimization
def thresholdOpt(preds_valid, y_valid):

    thresholds = np.linspace(0, 1, 50)
    ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in thresholds])

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
   
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
