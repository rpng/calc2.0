import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def rand_warp(images, out_size, max_warp=0.5, name='rand_hom'):
    num_batch = tf.shape(images)[0]
    y = tf.lin_space(-1., 1., 2)
    x = tf.lin_space(-1., 1., 2)
    py, px = tf.meshgrid(y, x)
    pts_orig = tf.tile(tf.concat([tf.reshape(px, [1, -1, 1]), 
                          tf.reshape(py, [1, -1, 1])],
                          axis=-1), [num_batch, 1, 1])
    x = pts_orig[:,:,0:1]
    y = pts_orig[:,:,1:2]

    rx1 = tf.random.uniform([num_batch, 2, 1], -1., -1.+ max_warp)
    rx2 = tf.random.uniform([num_batch, 2, 1], 1.- max_warp, 1.)
    rx = tf.concat([rx1, rx2], axis=1)
    
    ry1 = tf.random.uniform([num_batch, 2, 1], -1., -1.+max_warp)
    ry2 = tf.random.uniform([num_batch, 2, 1], 1.-max_warp, 1.)
    ry = tf.reshape(tf.concat([ry1, ry2], axis=2), [num_batch, 4, 1])

    pts_warp = tf.concat([rx, ry], axis=2)

    h = estimate_hom(pts_orig, pts_warp)
    return hom_warp(images, out_size, h)

def hom_warp(images, out_size, h, name='hom_warp'):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
            
            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _transform(images, out_size):
        with tf.variable_scope('_transform'):
            shape = tf.shape(images)
            num_batch = tf.shape(images)[0]
            num_channels = images.get_shape()[3]
            
            out_width = out_size[1]
            out_height = out_size[0]
            x = tf.linspace(-1., 1., out_width)
            y = tf.linspace(-1., 1., out_height)
            x, y = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.concat([tf.expand_dims(tf.reshape(x,[-1]),0),
                                  tf.expand_dims(tf.reshape(y,[-1]),0)], 0), 0)
            grid = tf.tile(grid, tf.stack([num_batch, 1, 1]))
            grid_hom = tf.concat([grid, tf.ones([num_batch, 1, tf.shape(grid)[-1]])], axis=1)
            
            W = tf.shape(images)[2]
            H = tf.shape(images)[1]
            W = tf.cast(W, tf.float32)
            H = tf.cast(H, tf.float32)
            
            #####################
            grid_warp = tf.matmul(h, grid_hom)
            grid_warp = grid_warp[:,:2,:] / grid_warp[:,2:3,:]
            
            x_s = grid_warp[:, 0, :]
            y_s = grid_warp[:, 1, :]

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                images, x_s_flat, y_s_flat,
                out_size)
            output = tf.reshape(input_transformed, 
                    tf.stack([num_batch, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(images, out_size)
        return output

def estimate_hom(src, dst):
    rx = src[:,:,0:1]
    ry = src[:,:,1:2]
    x = dst[:,:,0:1]
    y = dst[:,:,1:2]
    num_batch = tf.shape(src)[0]
    num_pts = tf.shape(src)[1]
    _0 = tf.zeros([num_batch, num_pts, 3])
    _1 = tf.ones([num_batch, num_pts, 1])
    A_even_rows = tf.concat([-rx, -ry, -_1, _0, rx*x, ry*x, x], axis=-1)
    A_odd_rows = tf.concat([_0, -rx, -ry, -_1, rx*y, ry*y, y], axis=-1)
    
    A = tf.concat([A_even_rows, A_odd_rows], axis=-1)
    A = tf.reshape(A, [num_batch, 2*num_pts, 9])
    _, _, V = tf.svd(A, full_matrices=True)
    return tf.reshape(V[:,:,-1], [num_batch, 3, 3])


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    from time import time
    from sys import argv
    im = cv2.cvtColor(cv2.imread(argv[1]),
            cv2.COLOR_BGR2RGB) / 255.
    h = im.shape[0]
    w = im.shape[1]
    x = tf.expand_dims(tf.placeholder_with_default(im.astype(np.float32), im.shape), 0)
    y = rand_warp(x, im.shape[:2])
    y = tf.clip_by_value(y+.5, 0.0, 1.)
    with tf.Session() as sess:
        t = time()
        p = sess.run(y)
        print("Took", 1000*(time()-t), "ms")
        plt.subplot(2,1,1)
        plt.imshow(im)
        plt.subplot(2,1,2)
        plt.imshow(np.squeeze(p))
        plt.show()



