from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib

import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
else:
    import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from calc2 import vh, vw, vss

import time
import traceback
import layers
from dataset.coco_classes import calc_classes, calc_class_names

gtdata = None
preddata = None
imdata = None
recdata = None


N_CLASSES = len(calc_classes.keys())

class CALC2(object):
    def __init__(self, model_dir, sess, 
            use_mu=False, ret_c_centers=False, ret_c5=False,
            checkpoint=None):
        assert not (ret_c_centers and ret_c5)
        self.sess = sess
        self.ret_c5 = ret_c5
        self.ret_c_centers = ret_c_centers
        self.images = tf.placeholder(tf.float32, [None, vh, vw, 3])    
        ret = vss(self.images, False, True, 
                ret_mu=use_mu, ret_c_centers=ret_c_centers,
                ret_c5=ret_c5)
        if ret_c5:
            self.descriptor = ret[0]
            self.c5 = ret[1]
        elif ret_c_centers:
            self.descriptor = ret[0]
            self.cc = ret[1]
        else:
            self.descriptor = ret if not use_mu else tf.reduce_mean(ret, axis=0, keepdims=True)

        saver = tf.train.Saver()
        if checkpoint is None:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            cpath = ckpt.model_checkpoint_path
        else:
            cpath = checkpoint
        print("loading model: ", cpath)
        saver.restore(self.sess, cpath)

    def run(self, images):
        
        if len(images.shape)==2:
            # Grayscale
            images = np.repeat(images[..., np.newaxis], 3, axis=-1)
        if len(images.shape)==3:
            images = images[np.newaxis, ...]

        if self.ret_c5:
            descr, c5 = self.sess.run(
                        [self.descriptor, self.c5],
                        feed_dict={self.images: images})
            return descr, c5
        elif self.ret_c_centers:
            descr, cc = self.sess.run(
                        [self.descriptor, self.cc],
                        feed_dict={self.images: images})
            return descr, cc
        else:
            descr = self.sess.run(self.descriptor, 
                        feed_dict={self.images: images})
            return descr

def kp_descriptor(tensor):
    
    b, h, w, c = tensor.shape
    assert b==1
    ky = []
    kx = []
    theta_full = []
    response_full = []
    n = 4
    for i in range(n):
        for j in range(n):
            _h = h//n
            _w = w//n
            _t = tensor[0, i*_h:(i+1)*_h, j*_w:(j+1)*_w]
            ky_, kx_ = np.unravel_index(np.argmax(
                    _t.reshape(-1, c), axis=0), (_h, _w)) 
            ky.append(ky_*(i+1))
            kx.append(kx_*(j+1))

            _t = np.pad(_t, ((1,1),(1,1),(0,0)), 'constant') 
            
            for k in range(len(ky_)):
                _ky = ky_[k] + 1 # +1 for pad
                _kx = kx_[k] + 1 
                _y = _t[_ky+1, _kx, k] - _t[_ky-1, _kx, k]
                _x = _t[_ky, _kx+1, k] - _t[_ky, _kx-1, k]
                theta_full.append(np.arctan2(_y, _x))
                response_full.append(_t[ky_[k], kx_[k], k])
            
    ky = np.concatenate(ky, axis=0)[...,np.newaxis]
    kx = np.concatenate(kx, axis=0)[...,np.newaxis]
    kp_full = np.concatenate((ky, kx), axis=1)
    kp = np.unique(kp_full, axis=0)
    # Keep unique kp with max activation
    mapping = {}
    for kp_i in kp:
        mapping[kp_i.tostring()] = np.where(kp_full == kp_i)[0]
    
    theta = np.empty((len(kp)), dtype=np.float32)
    response = np.empty((len(kp)), dtype=np.float32)
    for i in range(len(kp)):
        kp_i = kp
        inds = mapping[kp[i].tostring()]
        r = -np.inf
        t = -1
        for j in inds:
            if response_full[j] > r:
                r = response_full[j]
                t = theta_full[j]
        response[i] = r
        theta[i] = t
    ky = kp[:,0]
    kx = kp[:,1]

    ky = np.minimum(np.maximum(1, ky), h-2)
    kx = np.minimum(np.maximum(1, kx), w-2)

    kp_d = []
    kp = []
    pi = np.pi
 
    #tensor = np.pad(tensor, ((0,0),(1,1),(1,1),(0,0)), 'constant')
   
    for i in range(len(ky)):
        ky_i = ky[i] + 1
        kx_i = kx[i] + 1

        kp.append(cv2.KeyPoint(float(kx[i]), float(ky[i]),
                 _size=1.0, _response=10000*np.log(1+np.exp(response[i])), _angle=theta[i]))
        t = theta[i]
        od = [tensor[:, ky_i-1, kx_i-1],
                tensor[:, ky_i-1, kx_i],
                tensor[:, ky_i-1, kx_i+1],
                tensor[:, ky_i, kx_i-1],
                tensor[:, ky_i, kx_i+1],
                tensor[:, ky_i+1, kx_i-1],
                tensor[:, ky_i+1, kx_i],
                tensor[:, ky_i+1, kx_i+1]]
        
        _d = np.concatenate(tuple(od),
                axis=0).reshape(1,-1,c)

        d =  _d - tensor[:, ky_i, kx_i]
        d = d.reshape(-1,c) 
        kp_d.append(d.reshape(1,-1))
    kp_d = np.concatenate(kp_d, axis=0)
    return kp, kp_d

def show_example(image_fl, model_dir):
    im = cv2.cvtColor(cv2.resize(cv2.imread(image_fl), 
        (vw, vh)), cv2.COLOR_BGR2RGB)[np.newaxis, ...] / 255.0
    _im = tf.placeholder_with_default(im.astype(np.float32), im.shape)
    _, _, rec, seg, _, _, _ = vss(_im, False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rec, seg = sess.run([rec, seg])
        rec = (255*np.squeeze(rec)).astype(np.uint8)
        seg = np.argmax(np.squeeze(seg), axis=-1)
        rgb_seg = np.zeros(rec.shape).astype(np.uint8)
    
        np.random.seed(0)
        for i in range(N_CLASSES):
            c = np.random.rand(3)
            case = seg==i
            rgb_seg[case, :] = c
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        cv2.imwrite('plots/seg.jpg', rgb_seg)
        cv2.imwrite('plots/rec.jpg', rec)

def display_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        #print(variable.name)
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    print("\n\nTrainable Parameters: %d\n\n" % total_parameters)

def mask_helper(im, pred, rec, mask, title):
    h, w = pred.shape[:2]
    rgb1 = np.zeros((h, w, 3))
    rgb2 = np.zeros((h, w, 3))

    ones = np.ones((3))

    legend = []
    np.random.seed(0)
    for i in range(N_CLASSES):
        c = np.random.rand(3)
        case1 = mask==i
        case2 = pred==i
        if np.any(np.logical_or(case1, case2)):
            legend.append(Patch(facecolor=tuple(c), edgecolor=tuple(c),
                        label=calc_class_names[i]))

        rgb1[case1, :] = c
        rgb2[case2, :] = c   

    image1 = rgb1
    image2 = rgb2

    global imdata
    global preddata
    global recdata
    global gtdata

    if imdata is None:
        plt.subplot(2,2,1)
        imdata = plt.imshow(im)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])
        
        plt.subplot(2,2,2)
        recdata = plt.imshow(rec)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])


        plt.subplot(2,2,3)
        gtdata = plt.imshow(image1)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])

        plt.subplot(2,2,4)
        preddata = plt.imshow(image2)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])

    else:
        imdata.set_data(im)
        recdata.set_data(rec)
        gtdata.set_data(image1)
        preddata.set_data(image2)

    lgd = plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig = plt.gcf()
    fig.suptitle(title)
    
    plt.pause(1e-9)
    plt.draw()


def hard_neg_mine(descr):
    n = tf.shape(descr)[0]
    tlive = tf.tile(descr,
            [n, 1]) # [l0, l1, l2..., l0, l1, l2...]

    tmem = tf.reshape(tf.tile(tf.expand_dims(descr, 1),
            [1, n, 1]),
            [-1, descr.get_shape().as_list()[1]]) # [m0, m0, m0..., m1, m1, m1...]
    
    sim = tf.reduce_sum(tlive * tmem, axis=-1) # Cosine sim for rgb data + class data
    
    sim_sq = tf.reshape(sim,
        [n, n])
    
    # Make sure that this doesnt return the original descriptor
    # sim along diag is now -2 which is lower than cosine sim can go
    sim_sq = sim_sq - 3*tf.eye(n, dtype=tf.float32)

    # ID of nearest neighbor
    ids = tf.argmax(sim_sq, 
            axis=-1, output_type=tf.int32)

    # I guess just contiguously index it?
    row_inds = tf.range(0, n,
            dtype=tf.int32) * (n-1)
    buffer_inds = row_inds + ids
    #sim_nn = tf.nn.embedding_lookup(sim, buffer_inds)
    # Pull out the hard negative descriptors
    descr_n = tf.nn.embedding_lookup(tlive, buffer_inds)
    return descr_n

def log_msg(col_hdrs, row_hdr, values):
    msg = " "*(len(row_hdr)+2)
    for i in range(len(col_hdrs)):
        msg += "{0:^8s}".format(col_hdrs[i])

    msg += "\n" + " "*(len(row_hdr)+2)
    for i in range(len(col_hdrs)):
        msg += "{0:^8s}".format("-"*len(col_hdrs[i]))

    msg += "\n" + row_hdr + ": "
    for i in range(len(col_hdrs)):
        msg += "{0:^8.3f}".format(values[col_hdrs[i]])
    msg += "\n"
    print(msg)



class TrainingHook(tf.train.SessionRunHook):
    """A utility for displaying training information such as the loss, percent
    completed, estimated finish date and time."""

    def __init__(self, steps, eval_steps):
        self.steps = steps
        self.eval_steps = eval_steps
        self.last_time = time.time()
        self.last_est = self.last_time

        self.eta_interval = int(math.ceil(0.1 * self.steps))
        self.current_interval = 0

    def before_run(self, run_context):
        graph = tf.get_default_graph()
        runargs = {
            "loss": graph.get_collection("total_loss")[0],
            "segloss": graph.get_collection("segloss")[0],
            "recloss": graph.get_collection("recloss")[0],
            "simloss": graph.get_collection("simloss")[0],
            "kld": graph.get_collection("kld")[0],
            "im": graph.get_collection("im")[0],
            "pred": graph.get_collection("pred")[0],
            "rec": graph.get_collection("rec")[0],
            "label": graph.get_collection("label")[0],
        }

        return tf.train.SessionRunArgs(runargs)


    def after_run(self, run_context, run_values):
        step = run_context.session.run(tf.train.get_global_step())
        now = time.time()

        if self.current_interval < self.eta_interval:
            self.duration = now - self.last_est
            self.current_interval += 1
        if step % self.eta_interval == 0:
            self.duration = now - self.last_est
            self.last_est = now

        eta_time = float(self.steps - step) / self.current_interval * \
            self.duration
        m, s = divmod(eta_time, 60)
        h, m = divmod(m, 60)
        eta = "%d:%02d:%02d" % (h, m, s)

        if step % self.eval_steps == 0:

            im = run_values.results["im"] / 255.0
            pred = run_values.results["pred"] 
            rec = run_values.results["rec"] 
            mask = run_values.results["label"] 

            mask_helper(im, pred, rec, mask, "Train")
            tp = (step,
                  self.steps,
                  time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
                  eta,
                  run_values.results["loss"], 
                  run_values.results["segloss"],
                  run_values.results["recloss"],
                  run_values.results["simloss"],
                  run_values.results["kld"])

            print('\n(%d/%d): ETA: %s (%s)\n Train loss = %f, Seg = %f, Rec = %f, Sim = %f, KLD = %f' % tp)        

        self.last_time = now


class PredictionHook(tf.train.SessionRunHook):

    def __init__(self):
        pass
    
    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

class EvalHook(tf.train.SessionRunHook):
    """A utility for displaying training information such as the loss, percent
    completed, estimated finish date and time."""

    def __init__(self, savedir='model/plots', show=True, save=True):
        self.i = 0 
        self.show = show
        self.save = save
        self.savedir = savedir

        if not os.path.isdir(savedir):
            os.makedirs(savedir)                        

    def before_run(self, run_context):
        graph = tf.get_default_graph()
        runargs = {
            "loss": graph.get_collection("total_loss")[0],
            "segloss": graph.get_collection("segloss")[0],
            "recloss": graph.get_collection("recloss")[0],
            "simloss": graph.get_collection("simloss")[0],
            "kld": graph.get_collection("kld")[0],
            "im": graph.get_collection("im")[0],
            "pred": graph.get_collection("pred")[0],
            "rec": graph.get_collection("rec")[0],
            "label": graph.get_collection("label")[0],
        }
        return tf.train.SessionRunArgs(runargs)

    def after_run(self, run_context, run_values):

        step = run_context.session.run(tf.train.get_global_step())
        
        if self.i == 0:
            
            im = run_values.results["im"] / 255.0
            pred = run_values.results["pred"] 
            rec = run_values.results["rec"] 
            mask = run_values.results["label"] 

            mask_helper(im, pred, rec, mask, "Test")
            tp = (run_values.results["loss"], 
                  run_values.results["segloss"],
                  run_values.results["recloss"],
                  run_values.results["simloss"],
                  run_values.results["kld"])

            print('Test Error = %f, Seg = %f, Rec = %f, Sim = %f, KLD = %f' % tp)        
            fl = self.savedir + "/segmentation_iteration_%d.png" % step
            plt.savefig(fl, bbox_inches='tight', dpi=100)          

        self.i += 1

def standard_model_fn(func, steps, run_config, 
        optimizer_fn=None, eval_steps=32, model_dir='model'):
    """Creates model_fn for tf.Estimator.

    Args:
    func: A model_fn with prototype model_fn(features, labels, mode, hparams).
    steps: Training steps.
    run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
        synchronous training.
    optimizer_fn: The type of the optimizer. Default to Adam.

    Returns:
    model_fn for tf.estimator.Estimator.
    """

    def fn(features, labels, mode, params):
        """Returns model_fn for tf.estimator.Estimator."""

        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        ret = func(features, labels, mode, params)
        tf.add_to_collection("total_loss", ret["loss"])
        tf.add_to_collection("segloss", ret["segloss"])
        tf.add_to_collection("recloss", ret["recloss"])
        tf.add_to_collection("simloss", ret["simloss"])
        tf.add_to_collection("kld", ret["kld"])
        tf.add_to_collection("im", ret["im"])
        tf.add_to_collection("pred", ret["pred"])
        tf.add_to_collection("rec", ret["rec"])
        tf.add_to_collection("label", ret["label"])
        
        train_op = None

        training_hooks = []
        
        if is_training:

            plt.ion()

            training_hooks.append(TrainingHook(steps, eval_steps))

            if optimizer_fn is None:
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
            else:
                optimizer = optimizer_fn

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
            train_op = slim.learning.create_train_op(ret["loss"], optimizer)


        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=ret["predictions"],
            loss=ret["loss"],
            train_op=train_op,
            eval_metric_ops=ret["eval_metric_ops"],
            training_hooks=training_hooks,
            evaluation_hooks=[EvalHook(savedir=os.path.join(model_dir,'plots'))],
        )
    return fn


def num_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def train_and_eval(model_dir,
    steps,
    batch_size,
    model_fn,
    input_fn,
    hparams,
    log_steps=32,
    save_steps=1024,
    summary_steps=1024,
    eval_start_delay_secs=0,
    eval_throttle_secs=0):
    """Trains and evaluates our model. Supports local and distributed training.

    Args:
    model_dir: The output directory for trained parameters, checkpoints, etc.
    steps: Training steps.
    batch_size: Batch size.
    model_fn: A func with prototype model_fn(features, labels, mode, hparams).
    input_fn: A input function for the tf.estimator.Estimator.
    hparams: tf.HParams containing a set of hyperparameters.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved.
    save_checkpoints_secs: Save checkpoints every this many seconds.
    save_summary_steps: Save summaries every this many steps.
    eval_steps: Number of steps to evaluate model.
    eval_start_delay_secs: Start evaluating after waiting for this many seconds.
    eval_throttle_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago

    Returns:
    None
    """
    n_gpus = num_gpus()
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=n_gpus)

    run_config = tf.estimator.RunConfig(
      model_dir=model_dir,
      save_checkpoints_steps=save_steps,
      save_summary_steps=summary_steps,
      train_distribute=strategy,
      keep_checkpoint_max=None)
    
    estimator = tf.estimator.Estimator(
      model_dir=model_dir,
      model_fn=standard_model_fn(
          model_fn,
          steps,
          run_config,
          eval_steps=log_steps,
          model_dir=model_dir),
      params=hparams, config=run_config)
    
    train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(split="train", batch_size=batch_size),
      max_steps=steps)

    eval_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(split="validation", batch_size=batch_size),
      steps=100 // (batch_size // 3),
      start_delay_secs=eval_start_delay_secs,
      throttle_secs=eval_throttle_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def colored_hook(home_dir):
    """Colorizes python's error message.

    Args:
    home_dir: directory where code resides (to highlight your own files).
    Returns:
    The traceback hook.
    """

    def hook(type_, value, tb):
        def colorize(text, color, own=0):
            """Returns colorized text."""
            endcolor = "\x1b[0m"
            codes = {
              "green": "\x1b[0;32m",
              "green_own": "\x1b[1;32;40m",
              "red": "\x1b[0;31m",
              "red_own": "\x1b[1;31m",
              "yellow": "\x1b[0;33m",
              "yellow_own": "\x1b[1;33m",
              "black": "\x1b[0;90m",
              "black_own": "\x1b[1;90m",
              "cyan": "\033[1;36m",
            }
            return codes[color + ("_own" if own else "")] + text + endcolor

        for filename, line_num, func, text in traceback.extract_tb(tb):
            basename = os.path.basename(filename)
            own = (home_dir in filename) or ("/" not in filename)

            print(colorize("\"" + basename + '"', "green", own) + " in " + func)
            print("%s:  %s" % (
              colorize("%5d" % line_num, "red", own),
              colorize(text, "yellow", own)))
            print("  %s" % colorize(filename, "black", own))

        print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
    return hook
