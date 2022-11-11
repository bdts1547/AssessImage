#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace

import caffe


def get_histogram(resp, patches=10, f=np.sum):
    (i_filters, ih, iw) = resp.shape
    histogram = np.zeros((patches,patches,i_filters))
    patch_h = ih/float(patches)
    patch_w = iw/float(patches)

    for h in range(patches):
        for w in range(patches):
            ph = h*patch_h
            pw = w*patch_w
            patch_val = resp[:,int(ph):int(ph+patch_h), int(pw):int(pw+patch_w)]

            for b in range(i_filters):
                histogram[h,w,b] = f(patch_val[b])

    histogram_sum = np.sum(histogram, axis=2)
    normalized_histogram = histogram / histogram_sum[:,:,np.newaxis]

    return histogram, normalized_histogram


def get_symmetry(normalized_histogram_orig, normalized_histogram_flip):
    assert(normalized_histogram_orig.shape == normalized_histogram_flip.shape)
    sum_abs = np.sum(np.abs(normalized_histogram_orig - normalized_histogram_flip))
    sum_max = np.sum(np.maximum(normalized_histogram_orig, normalized_histogram_flip))
    return 1.0 - sum_abs / sum_max


def load_model(model):
    if model == "alexnet":
        model_weights = os.path.join('model_symmetry/bvlc_alexnet/bvlc_alexnet.caffemodel')
        model_model = os.path.join('model_symmetry/bvlc_alexnet/deploy.prototxt')
        net_mean = np.load('model_symmetry/ilsvrc_2012_mean.npy').mean(1).mean(1)
    elif model == "alexnet512":
        model_weights = os.path.join('model_symmetry/bvlc_alexnet/bvlc_alexnet.caffemodel')
        model_model = os.path.join('model_symmetry/bvlc_alexnet/deploy-512.prototxt')
        net_mean = np.load('model_symmetry/ilsvrc_2012_mean.npy').mean(1).mean(1)
    elif model == "alexnet600x800":
        model_weights = os.path.join('model_symmetry/bvlc_alexnet/bvlc_alexnet.caffemodel')
        model_model = os.path.join('model_symmetry/bvlc_alexnet/deploy-600x800.prototxt')
        net_mean = np.load('model_symmetry/ilsvrc_2012_mean.npy').mean(1).mean(1)
    else:
        assert False, "Unknown model"

    if False:
        caffe.set_mode_gpu()
        caffe.set_device(0)

    null_fds = os.open(os.devnull, os.O_RDWR)
    out_orig = os.dup(2)
    os.dup2(null_fds, 2)
    net = caffe.Net(model_model, model_weights, caffe.TEST)
    os.dup2(out_orig, 2)
    os.close(null_fds)

    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_mean("data", net_mean) # imagenet mean (bgr)
    transformer.set_channel_swap("data", (2,1,0))
    transformer.set_transpose("data", (2,0,1))
    transformer.set_raw_scale("data", 255)
    
    return net, transformer

#-------------------------------------------------------------------------------

def run_process_images(args, files_dict, callback=None):
    # print("processing symmetry")
    MODEL, layer = "alexnet512", "conv1"
    filter_filters = None

 
    if args.var_run_symmetry:
        # print( "Symmetry", args.var_args_symmetry)
        result_names = ["symmetry-lr", "symmetry-ud", "symmetry-lrud"]
        tok = args.var_args_symmetry.split()
        if len(tok)!=3:
            return False
        MODEL, layer, patches = tok[0], tok[1], int(tok[2])
    else:
        assert False

    net, transformer = load_model(MODEL)

    if ":" in layer:
        layer, filter_filters = layer.split(":")
        filters = net.params[layer][0].data
        colorfulness = np.zeros(filters.shape[0])
        for i in range(filters.shape[0]):
            colorfulness[i] = np.sum(np.std(filters[i], axis=0))
        #colorfulness /= np.max(colorfulness)
        colorfulness_thres = colorfulness[colorfulness.argsort()][96/2]
        if filter_filters=="bw":
            colorfulness = colorfulness < colorfulness_thres # bw
        elif filter_filters=="co":
            colorfulness = colorfulness >= colorfulness_thres # color
        else:
            return False
        #np.set_printoptions(precision=3)
        # print( colorfulness)

   
    num_k = len(files_dict)
    for count_k, k in enumerate(files_dict):

        log = open(args.var_saveto, 'w')

        num_f = len(files_dict[k])
        for count_f, f in enumerate(files_dict[k]):
            img_name = os.path.join(args.var_root, k, f)
            

            print(img_name, args.var_root, k, f)
            img = caffe.io.load_image(img_name)
            source_img = transformer.preprocess('data',img)[None,:]
            net.forward(data = source_img)
            resp = net.blobs[layer].data[0]
            if not filter_filters is None:
                resp = resp[colorfulness]
                # print( "SHAPE", resp.shape)
            vals = []

        
            if args.var_run_symmetry:
                histogram8_orig,normalized_histogram8_orig = get_histogram(resp, patches=patches, f=np.max)

                source_img = transformer.preprocess('data',np.fliplr(img))[None,:]
                net.forward(data = source_img)
                resp = net.blobs[layer].data[0]
                histogram8_fliplr,normalized_histogram8_fliplr = get_histogram(resp, patches=patches, f=np.max)

                source_img = transformer.preprocess('data',np.flipud(img))[None,:]
                net.forward(data = source_img)
                resp = net.blobs[layer].data[0]
                histogram8_flipud,normalized_histogram8_flipud = get_histogram(resp, patches=patches, f=np.max)

                source_img = transformer.preprocess('data',np.fliplr(np.flipud(img)))[None,:]
                net.forward(data = source_img)
                resp = net.blobs[layer].data[0]
                histogram8_fliplrud,normalized_histogram8_fliplrud = get_histogram(resp, patches=patches, f=np.max)

                lr = get_symmetry(histogram8_orig, histogram8_fliplr)
                ud = get_symmetry(histogram8_orig, histogram8_flipud)
                lrud = get_symmetry(histogram8_orig, histogram8_fliplrud)

                # print(lr, ud, lrud)

                vals.append(lr)
                vals.append(ud)
                vals.append(lrud)
        
            else:
                assert False

            print(vals)
            print("Write")
            if args.var_run_symmetry:
                log.write(os.path.basename(f)+","+",".join([str(r) for r in vals])+"\n")



    return True


def get_files( var_root):
        categories, files = [], {}
        var_root = unicode(var_root)
        for category in os.listdir(var_root):
          absolute = os.path.join(var_root, category)
          if os.path.isdir(absolute):
            categories.append(category)
            files[category] = []
            for img in os.listdir(absolute):
              name, ext = os.path.splitext(img)
              if ext.lower() in [".png",".tif",".jpg",".jpeg"] and name[0]!=".":
                files[category].append(img)
            files[category] = sorted(files[category])
        return files


def getConfig():
    # print('sss')
    parser = Namespace()
    parser.var_root='data'
    parser.var_run_symmetry=True
    parser.var_saveto='score_symmetry.csv'
    parser.var_args_symmetry = 'alexnet512 conv1 17'

    return parser


def run():
    args = getConfig()
    # print(args)

    files_dict = get_files(args.var_root)
    num_files = sum([len(files_dict[k]) for k in files_dict])
   
    
    # print "Files_dict: %s" % files_dict
    # print "Saving to: %s" % args["var_saveto"]
    # print "Root Directory: %s" % args.var_root
    # for k in files_dict:
    #     print "%4d %s" % (len(files_dict[k]), k)
    # print args

    if run_process_images(args, files_dict):
        # step_callback("done")
        # print("Done symmetry")
        return True
    else:
        # step_callback("error")
        # print("Error while running symmetry!")
        return False 
        



if __name__ == '__main__':
    run()