#!/usr/bin/env python3

from __future__ import print_function
import cv2
from os.path import join
import numpy as np
import utils
import calc2
from time import time
import tensorflow as tf

vh = calc2.vh
vw = calc2.vw
K = 7
N = 200
C = 7
W = 9

def close_loop(db, dbkp, descr, kp):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    kp, kp_d = kp
    db = np.concatenate(tuple(db), axis=0)
    sim = np.sum(descr * db, axis=-1) 
    
    top_k_sim_ind = np.argpartition(sim, -K)[-K:] 

    max_sim = -1.0
    i_max_sim = -1
    best_match_tuple = None

    for k in top_k_sim_ind:
        db_kp, db_kp_d = dbkp[k]
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
    return i_max_sim
    
def kitti_loop(gui, vo_fn, im_dir, seq):	
    i = 0
    loops = []
    db = []
    dbkp = []
    avg_rate = 0
    loop_count = 0
    last_loop_id = -1
    skipped = False
    j = 0

    with open(vo_fn, 'r') as vo_f, open('kitti_traj.txt', 'w') as t_f, \
            open('kitti_loops.txt', 'w') as l_f, tf.Session() as sess:
        
        calc = utils.CALC2('model', sess, ret_c5=True)
        qt = []
        pts = []
        ims = []
        for line in vo_f.readlines():
            if len(line) != 0 and line[0] != "#": # skip comments and empty line at the end
                line_split = line.split()
                frame_id = str(i)
                i += 1
                x = line_split[3]
                y = line_split[7]
                pts.append([float(x), float(y)])
                t_f.write(frame_id + ',' + x + ',' + y + '\n')
     
                fl_nm = str(i).zfill(6) + ".png"
                im = cv2.imread(join(im_dir, "image_2/"+ fl_nm))
                im_cp = np.copy(im)
                ims.append(im_cp)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                if im is None or im.shape[0] == 0:
                    print("No image: %s" % fl_nm)
                    break
                
                w = int(4.0 / 3 * im.shape[0]) // 2
                _w = im.shape[1] // 2
                #im = im[:, (_w-w):(_w+w+1),:]
                im = cv2.resize(im, (vw, vh))
                t0 = time()
                
                descr, c5 = calc.run(im)
                kp, kp_d = utils.kp_descriptor(c5)
                dbkp.append((kp, kp_d))
                db.append(descr)       
                if i > 2*N:           
                    t1 = time()
                    j = close_loop(db[:-N], dbkp, descr, (kp, kp_d))
                    t = (time() - t1) * 1000
                    qt.append(str(len(db)) + "," + str(t) + "\n")     
                    if j > 0:
                        if last_loop_id == -1 or loop_count == 0:
                            last_loop_id = j
                            print("LCD HYPOTHESIS: %d -> %d" % (i,j))
                            loop_count += 1
                        elif abs(j - last_loop_id) < W:
                            print("LCD HYPOTHESIS INCREASE: %d -> %d" % (i,j))
                            loop_count += 1
                        else:
                            loop_count = 0
                            last_loop_id = -1
                            skipped = False
                    else:
                        loop_count = 0
                        last_loop_id = -1
                        skipped = False
                # Only time processing, not IO
                if i > 0: # TF take one run to warm up. Dont fudge results
                    rate = 1.0 / (time()-t0)
                    avg_rate += (rate - avg_rate) / (i+1)
                    print("Frame %d, rate = %f Hz, avg rate = %f Hz" % (i, rate, avg_rate))
                
                is_loop = False
                loop_im = None
                if loop_count >= C:
                    is_loop = True
                    print("LOOP DETECTED: %d -> %d" % (i,j))
                    ii = len(pts)-C//2-1
                    jj = j-W//2
                    l_f.write(str(pts[ii][0]) + "," + \
                        str(pts[ii][1]) + "," + str(pts[jj][0]) + \
                        "," + str(pts[jj][1]) + "\n")
                    loop_count = 0
                    skipped = False
                    match = np.concatenate((ims[ii], ims[jj]), axis=1)
                    cv2.imwrite('plots/match_kitti%s_%d_%d.png' % (seq, ii, jj), match)
                    # remove loop descr since we have revisited this location
                    #db = db[:jj] + db[jj+W//2:]
                    #pts = pts[:jj] + pts[jj+W//2:]
                    loop_im = ims[jj]
                    # ims = ims[:jj] + ims[jj+W//2:]

                gui.update(float(x), float(y), is_loop, im_cp, loop_im)
                

    with open('kitti_q_times.txt', 'w') as q_f:
        q_f.writelines(qt)

class Display:
    def __init__(self):
        self.w = 2*640
        self.h = 2*480
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.x_bounds = [-50.0, 10.0]
        self.y_bounds = [-7.0, 7.0]
        self.xs = [0.]
        self.ys = [0.]
        self.match = False

    def world2canvas(self, x, y):
        x_bounds = self.x_bounds
        y_bounds = self.y_bounds
        new_x = int(self.w*(x - x_bounds[0])/(x_bounds[1] - x_bounds[0]))
        new_y = int(self.h*(y - y_bounds[0])/(y_bounds[1] - y_bounds[0]))
        return new_x, new_y

    def update(self, x, y, is_loop, _im, _loop_im = None):
        x0, y0 = self.world2canvas(self.xs[-1], self.ys[-1])
        x1, y1 = self.world2canvas(x, y)
        cv2.line(self.canvas, (x0,y0), (x1,y1), color=(255,0,0), thickness=7)
        self.xs.append(x)
        self.ys.append(y)

        sc = .4
        h,w = (int(sc*_im.shape[0]),int(sc*_im.shape[1]))
        im = cv2.resize(_im, (w,h))
        self.canvas[100:(100+h), :w, :] = im
        
        cv2.putText(self.canvas, "Query", (0,50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255))
        cv2.putText(self.canvas, "Match", (0,150+h), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255))

        if is_loop:
            r = 7
            cv2.circle(self.canvas, (x1,y1), r, color=(0,0,255), thickness=-1)
            loop_im = cv2.resize(_loop_im, (w,h))
            self.match = True
            self.canvas[(200+h):(200+2*h), :w, :] = loop_im
        elif self.match == False:
            loop_im = 30*np.ones((h,w,3),dtype=np.uint8)
            cv2.putText(loop_im, "No Match", (66,h//2), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255,255,255))
            self.canvas[(200+h):(200+2*h), :w, :] = loop_im

        cv2.imshow("Loops", self.canvas)
        cv2.waitKey(1)

if __name__ == '__main__':

    data_root = "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/kitti_odom/"
    seq = "6"
    vo_fn = data_root + "dataset/poses/" + seq.zfill(2) + ".txt"
    im_dir = data_root + "dataset/sequences/" + seq.zfill(2) 

    gui = Display()

    kitti_loop(gui, vo_fn, im_dir, seq)


