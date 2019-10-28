
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt, rcParams
from matplotlib.font_manager import FontProperties
import cv2
from sklearn.metrics import precision_recall_curve, auc
from time import time
import re
from os import path, getcwd, listdir, makedirs
import sys
from dataset.coco_classes import calc_classes

try:
    import caffe
except:
    print("Cannot impport caffe. Do not run with --include_calc until caffe is installed.")

import utils
from utils import kp_descriptor
import calc2

def smooth_pr(prec, rec):
    return prec, rec
    n = len(prec)
    m = 11
    p_smooth = np.zeros((m), dtype=np.float)
    r_smooth = np.linspace(0.0, 1.0, m) 
    for i in range(m):
        j = np.argmin( np.absolute(r_smooth[i] - rec) ) + 1
        p_smooth[i] = np.max( prec[:j] )
    return p_smooth, r_smooth

def check_match(im_lab_k, db_lab, num_include):
    return (num_include == 1 and db_lab==im_lab_k) or \
        ((db_lab-num_include/2) <= im_lab_k and im_lab_k <= (db_lab+num_include/2))

def get_prec_recall(model_dir, data_path, num_include=5,
        title='Precision-Recall Curve', checkpoint=None,
        netvlad_feat=None, include_calc=False):
                
    database = [] # stored descriptors
    database_kp = [] # stored kp and kp descriptors
    database_labels = [] # the image labels    
    database_ims = []
 
    db_calc1 = []   

    mem_path = data_path + "/memory"
    live_path = data_path + "/live"

    print("memory path: ", mem_path)
    print("live path: ", live_path)

    mem_files = sorted([path.join(mem_path, f) for f in listdir(mem_path)])
    live_files = sorted([path.join(live_path, f) for f in listdir(live_path)])
    
    if netvlad_feat is not None:
        # Assumes netvlad file order was sorted too
        db_netvlad = np.fromfile(netvlad_feat + "_db.bin",
                         dtype=np.float32).reshape(len(mem_files), -1)
        
        q_netvlad = np.fromfile(netvlad_feat + "_q.bin",
                         dtype=np.float32).reshape(len(live_files), -1)
    t_calc = []
    ims = np.empty((1, calc2.vh, calc2.vw, 3), dtype=np.float32)

    if include_calc:
        caffe.set_mode_gpu()
        # Must be placed in 'calc_model'
        calc1 = caffe.Net('calc_model/deploy.prototxt', 1, weights='calc_model/calc.caffemodel')
        # Use caffe's transformer
        transformer = caffe.io.Transformer({'X1':(1,1,120,160)})	
        transformer.set_raw_scale('X1',1./255)
	
    n_incorrect = 0

    with tf.Session() as sess:
        calc = utils.CALC2(model_dir, sess, ret_c5=True, checkpoint=checkpoint)
        for fl in mem_files:    
            print("loading image ", fl, " to database")
            #ims[0] = cv2.cvtColor(cv2.resize(cv2.imread(fl), (calc2.vw, calc2.vh)), 
            #            cv2.COLOR_BGR2RGB)
            _im = cv2.imread(fl)
            im = cv2.cvtColor(cv2.resize(_im,
                (calc2.vw, calc2.vh)), cv2.COLOR_BGR2RGB)
            database_ims.append(im)
            ims[0] = im / 255.0
            t0 = time()
            descr, c5 = calc.run(ims)
            t_calc.append(time()-t0)
            kp, kp_d = kp_descriptor(c5)
            database_kp.append((kp, kp_d))
            database.append(descr)       
            database_labels.append(int(re.match('.*?([0-9]+)$',
                path.splitext(path.basename(fl))[0]).group(1)))
           
            if include_calc: 
                im = cv2.equalizeHist(cv2.cvtColor(cv2.resize(_im, 
                        (160, 120), interpolation = cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2GRAY))
                calc1.blobs['X1'].data[...] = transformer.preprocess('X1', im)
                calc1.forward()
                d = np.copy(calc1.blobs['descriptor'].data[...])
                d /= np.linalg.norm(d)
                db_calc1.append(d)

        correct = []
        scores = []
        
        correct_reg = []
        scores_reg = []
        
        correct_nv = []
        scores_nv = []
        
        correct_c1 = []
        scores_c1 = []

        if include_calc:
            db_calc1 = np.concatenate(tuple(db_calc1), axis=0)

        database = np.concatenate(tuple(database), axis=0)
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        #plt.ion()
        imdata = None
        i = 0
        for fl in live_files:    
            im_label_k = int(re.match('.*?([0-9]+)$', 
                    path.splitext(path.basename(fl))[0]).group(1))
             
            _im = cv2.imread(fl)
            im = cv2.cvtColor(cv2.resize(_im,
                (calc2.vw, calc2.vh)), cv2.COLOR_BGR2RGB)
            ims[0] = im / 255.0
            t0 = time()
            descr, c5 = calc.run(ims)
            t_calc.append(time()-t0)
            kp, kp_d = kp_descriptor(c5)
            
            if netvlad_feat is not None:
                sim_nv = np.sum(q_netvlad[i:i+1,:] * db_netvlad, axis=-1) 
                i_max_sim_nv = np.argmax(sim_nv)
                max_sim_nv= sim_nv[i_max_sim_nv]
                scores_nv.append(max_sim_nv)
                db_lab = database_labels[i_max_sim_nv]  
                correct_nv.append(int(check_match(im_label_k, db_lab, num_include)))
            
            if include_calc:
                im = cv2.equalizeHist(cv2.cvtColor(cv2.resize(_im, 
                        (160, 120), interpolation = cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2GRAY))
                calc1.blobs['X1'].data[...] = transformer.preprocess('X1', im)
                calc1.forward()
                d = np.copy(calc1.blobs['descriptor'].data[...])
                d /= np.linalg.norm(d)
                sim_c1 = np.sum(d * db_calc1, axis=-1) 
                
                i_max_sim_c1 = np.argmax(sim_c1)
                max_sim_c1 = sim_c1[i_max_sim_c1]

            sim = np.sum(descr * database, axis=-1) 
            
            i_max_sim_reg = np.argmax(sim)
            max_sim_reg = sim[i_max_sim_reg]
            
            t0 = time()
            K = 7
            top_k_sim_ind = np.argpartition(sim, -K)[-K:] 

            max_sim = -1.0
            i_max_sim = -1
            best_match_tuple = None
            for k in top_k_sim_ind:
                db_kp, db_kp_d = database_kp[k]
                matches = matcher.knnMatch(kp_d, db_kp_d, 2)
                good = []
                pts1 = []
                pts2 = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                        pts1.append(db_kp[m.trainIdx].pt)
                        pts2.append(kp[m.queryIdx].pt)
                if len(good) > 7:
                    pts1 = np.int32(pts1)
                    pts2 = np.int32(pts2)
                    curr_sim = sim[k]
                    if curr_sim > max_sim:
                        max_sim = curr_sim
                        i_max_sim = k
                        best_match_tuple = (kp, db_kp, good, pts1, pts2)
            
            if i_max_sim > -1:
                F, mask = cv2.findFundamentalMat(best_match_tuple[3],
                             best_match_tuple[4], cv2.FM_RANSAC)
                if F is None:
                    max_sim=-1.0
                    i_max_sim = -1
            print("Comparison took", (time()-t0)*1000, "ms")
            scores.append(max_sim)
            db_lab = database_labels[i_max_sim]  
            correct.append(int(check_match(im_label_k, db_lab, num_include)) if i_max_sim > -1 else 0)
            scores_reg.append(max_sim_reg)
            db_lab = database_labels[i_max_sim_reg]  
            correct_reg.append(int(check_match(im_label_k, db_lab, num_include)))
            
            if include_calc:           
                scores_c1.append(max_sim_c1)
                db_lab = database_labels[i_max_sim_c1]  
                correct_c1.append(int(check_match(im_label_k, db_lab, num_include)))
            '''
            if correct[-1]:
                mask = np.squeeze(mask)
                good = []
                for i in range(len(best_match_tuple[2])):
                    if mask[i]:
                        good.append(best_match_tuple[2][i])
                if 1: #imdata is None:
                    imdata = plt.imshow(cv2.drawMatches(im, best_match_tuple[0], 
                        database_ims[i_max_sim], best_match_tuple[1], good,
                        None, flags=4))
                else:
                    imdata.set_data(cv2.drawMatches(im, best_match_tuple[0], 
                        database_ims[i_max_sim], best_match_tuple[1], good,
                        None, flags=4))
                plt.pause(0.0000001)
                plt.show()
            '''
            print("Proposed match G-CALC2:", im_label_k, ", ", 
                    database_labels[i_max_sim], ", score = ", 
                    max_sim, ", Correct =", correct[-1])
            print("Proposed match CALC2:", im_label_k, ", ", 
                    database_labels[i_max_sim_reg], ", score = ", 
                    max_sim_reg, ", Correct =", correct_reg[-1])
            if include_calc:
                print("Proposed match CALC:", im_label_k, ", ", 
                        database_labels[i_max_sim_c1], ", score = ", 
                        max_sim_c1, ", Correct =", correct_c1[-1])
            if netvlad_feat is not None:
                print("Proposed match NetVLAD:", im_label_k, ", ", 
                        database_labels[i_max_sim_nv], ", score = ", 
                        max_sim_nv, ", Correct =", correct_nv[-1])
            print()
            i += 1
    print("Mean CALC2 run time: %f ms" % (1000 * np.mean(np.array(t_calc))))

    precision, recall, threshold = precision_recall_curve(correct, scores)
    
    precision_reg, recall_reg, threshold = precision_recall_curve(correct_reg, scores_reg)
    
    precision_c1 = recall_c1 = None
    if include_calc:
        precision_c1, recall_c1, threshold = precision_recall_curve(correct_c1, scores_c1)

    pnv = rnv = None
    if netvlad_feat is not None:
        pnv, rnv, threshold = precision_recall_curve(correct_nv, scores_nv)
    print("N Incorrect:", n_incorrect)

    return precision, recall, precision_reg, recall_reg, precision_c1, recall_c1, pnv, rnv


def plot(model_dir, data_path, num_include=5, 
        title='Precision-Recall Curve', netvlad_feat=None,
        include_calc=False):
    
    t0 = time()

    precision, recall, precision_reg, recall_reg, precision_c1,\
            recall_c1, pnv, rnv = get_prec_recall(model_dir, data_path,
                    num_include, title, netvlad_feat=netvlad_feat,
                    include_calc=include_calc)

    rcParams['font.sans-serif'] = 'DejaVu Sans'
    rcParams['font.weight'] = 'bold'
    rcParams['font.size'] = 12
    rcParams['axes.titleweight'] = 'bold'    
    rcParams['axes.labelweight'] = 'bold'    
    rcParams['axes.labelsize'] = 'large'    
    rcParams['figure.figsize'] = [8.0, 4.0]    
    rcParams['figure.subplot.bottom'] = 0.2    
    rcParams['savefig.dpi'] = 200.0
    rcParams['figure.dpi'] = 200.0
    plots = path.join(getcwd(), "plots")
    rcParams['savefig.directory'] = plots
    if not path.isdir(plots):
        makedirs(plots)
    
    ax = plt.gca()
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1.01])
    plt.xlim([0, 1])
    plt.title(title)
    
    handles = []
    
    perf_prec = abs(precision - 1.0) <= 1e-6
    r = 0
    if np.any(perf_prec):	
        r = np.max(recall[perf_prec]) 
    h, = ax.plot(recall, precision, '-',
            label='G-CALC2 (AUC=%0.2f, r=%0.2f)' % (auc(recall, precision), r),
            linewidth=1.5, color='b')
    handles.append(h)

    perf_prec = abs(precision_reg - 1.0) <= 1e-6
    r = 0
    if np.any(perf_prec):	
        r = np.max(recall_reg[perf_prec]) 
    h, = ax.plot(recall_reg, precision_reg, '--', 
            label='CALC2 (AUC=%0.2f, r=%0.2f)' % (auc(recall_reg, precision_reg), r),
            linewidth=1.5, color='b')
    handles.append(h)
    
    if include_calc:
        perf_prec = abs(precision_c1 - 1.0) <= 1e-6
        r = 0
        if np.any(perf_prec):	
            r = np.max(recall_c1[perf_prec]) 
        h, = ax.plot(recall_c1, precision_c1, '-', 
                label='CALC (AUC=%0.2f, r=%0.2f)' % (auc(recall_c1, precision_c1), r),
                linewidth=1.5, color='g')
        handles.append(h)
    
    if netvlad_feat is not None:
        perf_prec = abs(pnv - 1.0) <= 1e-6
        r = 0
        if np.any(perf_prec):	
            r = np.max(rnv[perf_prec]) 
        h, = ax.plot(rnv, pnv, '-.', 
                label='NetVLAD (AUC=%0.2f, r=%0.2f)' % (auc(rnv, pnv), r),
                linewidth=1.5, color='r')
        handles.append(h)

    fontP = FontProperties()
    fontP.set_size('small')
    leg = ax.legend(handles=handles, fancybox=True, loc='best',
             prop=fontP, handlelength=3)
    leg.get_frame().set_alpha(0.5) # transluscent legend :D
    leg.set_draggable(True)
    for line in leg.get_lines():
        line.set_linewidth(3)
    
    print("Elapsed time = ", time()-t0, " sec")
    plt.show()    

def find_best_checkpoint(model_dir, data_path, num_include=5):
    ckpt_files = [os.path.join(model_dir, f.split('.meta')[0])  
        for f in os.listdir(model_dir) if os.path.isfile(
        os.path.join(model_dir, f)) and 'meta' in f]

    best_cp = ''
    best_auc = -1
    best_cp_reg = ''
    best_auc_reg = -1
    for ckpt_file in ckpt_files:
        precision, recall, pr,  rr = get_prec_recall(model_dir, data_path,
            num_include, '', checkpoint=ckpt_file)
        tf.reset_default_graph()
        cauc = auc(recall, precision)
        if cauc > best_auc:
            best_auc = cauc
            best_cp = ckpt_file
        cauc = auc(rr, pr)
        if cauc > best_auc_reg:
            best_auc_reg = cauc
            best_cp_reg = ckpt_file


    print("checkpoint:", best_cp, ", AUC =", best_auc)
    print("Reg checkpoint:", best_cp_reg, ", AUC =", best_auc_reg)

def show_local_descr(model_dir, im_fls, train_dirs, cls):
    vh = calc2.vh
    vw = calc2.vw
    im_fls = im_fls.split(',')
    assert len(im_fls)==3
    
    train_fls = []
    for i in range(len(train_dirs)):
        for f in listdir(train_dirs[i]): 
            train_fls.append(path.join(train_dirs[i], f))

    from sklearn.decomposition import KernelPCA as PCA
    import matplotlib.patches as mpatches

    N = 2

    train_ims = np.empty((len(train_fls),vh,vw,3), dtype=np.float32)
    for i in range(len(train_fls)):
        train_ims[i] = cv2.cvtColor(cv2.resize(cv2.imread(train_fls[i]), (vw,vh)), 
                    cv2.COLOR_BGR2RGB) / 255.

    ims = np.empty((3,vh,vw,3), dtype=np.float32)
    for i in range(len(im_fls)):
        ims[i] = cv2.cvtColor(cv2.resize(cv2.imread(im_fls[i]), (vw,vh)), 
                    cv2.COLOR_BGR2RGB) / 255.

    with tf.Session() as sess:
        calc = utils.CALC2(model_dir, sess)
        
        d_train = calc.run(train_ims).reshape((len(train_fls),
                            vh//16*vw//16,4*(1+len(calc_classes.keys()))))
        # Each class has 4 local descriptors
        didx = 4*(1+calc_classes[cls[0]])
        d_cls_train = d_train[:, :, didx:didx+4].reshape(4*len(train_fls),-1)
        didx2 = 4*(1+calc_classes[cls[1]])
        d_cls2_train = d_train[:, :, didx2:didx2+4].reshape(4*len(train_fls),-1)
        d_app_train = d_train[:, :, :4].reshape(4*len(train_fls),-1)

        d = calc.run(ims).reshape((3,vh//16*vw//16,4*(1+len(calc_classes.keys()))))

        # Now just take the first local descriptor
        d_cls = d[:, :, didx:didx+1].reshape(3,-1)
        d_cls2 = d[:, :, didx2:didx2+1].reshape(3,-1)
        d_app = d[:, :, :1].reshape(3,-1)
        
    pca = PCA(N)
    pca.fit(d_cls_train)
    dcc1 = pca.transform(d_cls) # calculate the principal components
    dcc1 = dcc1 / np.linalg.norm(dcc1, axis=-1)[...,np.newaxis]    

    pca.fit(d_cls2_train)
    dcc2 = pca.transform(d_cls2) # calculate the principal components
    dcc2 = dcc2 / np.linalg.norm(dcc2, axis=-1)[...,np.newaxis]    
    
    pca.fit(d_app_train)
    dac = pca.transform(d_app) # calculate the principal components
    dac = dac / np.linalg.norm(dac, axis=-1)[...,np.newaxis]    
    
    minx = -1.1 #min(np.min(dac[:,0]), np.min(dcc1[:,0]))-.1
    maxx = 1.1 # max(np.max(dac[:,0]), np.max(dcc1[:,0]))+.1
    miny = -1.1 #min(np.min(dac[:,1]), np.min(dcc1[:,1]))-.1
    maxy = 1.1 #max(np.max(dac[:,1]), np.max(dcc1[:,1]))+.1
    x = np.zeros_like(dac[:,0])
    
    rcParams['font.sans-serif'] = 'DejaVu Sans'
    rcParams['font.size'] = 10
    rcParams['patch.linewidth'] = .5 
    rcParams['figure.figsize'] = [8.0, 3.0]    
    rcParams['figure.subplot.bottom'] = 0.2    
    rcParams['savefig.dpi'] = 200.0
    rcParams['figure.dpi'] = 200.0

    fig = plt.figure()
    ax = fig.add_subplot(131, aspect='equal')
    ax.quiver(x, x, dcc1[:,0], dcc1[:,1], color=['b','g','r'], 
            scale=1, units='xy', width=.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    plt.title(cls[0])
    
    ax = fig.add_subplot(132, aspect='equal')
    ax.quiver(x, x, dcc2[:,0], dcc2[:,1], color=['b','g','r'], 
        scale=1, units='xy', width=.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    plt.title(cls[1])
    
    ax = fig.add_subplot(133, aspect='equal')
    ax.quiver(x, x, dac[:,0], dac[:,1], color=['b','g','r'], 
        scale=1, units='xy', width=.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    plt.title('appearance')
    
    l1 = mpatches.Patch(color='b', label='database')
    l2 = mpatches.Patch(color='g', label='positive')
    l3 = mpatches.Patch(color='r', label='negative')
    h = plt.legend(handles=[l1, l2, l3])
    h.get_frame().set_alpha(0.0) # transluscent legend :D
    h.set_draggable(True)
    plt.show()    

if __name__=='__main__':
    
    show_local_descr('model',
         'dataset/CampusLoopDataset/memory/Image092.jpg,' + \
         'dataset/CampusLoopDataset/live/Image090.jpg,' + \
         'dataset/CampusLoopDataset/live/Image037.jpg',
         ["dataset/CampusLoopDataset/memory", "dataset/CampusLoopDataset/live"],
         ['wall', 'structure-other'])
