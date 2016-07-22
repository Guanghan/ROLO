"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
This is a modified version of SORT, intended for single object visual tracking.
It outputs the tracking results of [YOLO + kalman filter]

Guanghan Ning
gnxr9@mail.missouri.edu
"""

from __future__ import print_function

import scipy
print(scipy.version)
print(scipy.version.version)


#from numba import jit
from filterpy.kalman import KalmanFilter
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse


#@jit
def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/h
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the form [x,y,s,r] and returns it in the form
    [x1,y1,x2,x2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:]) 
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

def choose_video_sequence(test):

    # For VOT-30:
    if test == 0:
        w_img, h_img = [480, 640]
        sequence_name = 'Human2'
        training_iters = 250
        testing_iters = 1128
    elif test == 1:
        w_img, h_img = [320, 240]
        sequence_name = 'Human9'
        training_iters = 70
        testing_iters = 302
    elif test == 2:
        w_img, h_img = [320, 240]
        sequence_name = 'Suv'
        training_iters = 314
        testing_iters = 943
    elif test == 3:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurBody'
        training_iters = 111
        testing_iters = 334
    elif test == 4:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurCar1'
        training_iters = 247
        testing_iters = 742#988
    elif test == 5:
        w_img, h_img = [352, 240]
        sequence_name = 'Dog'
        training_iters = 42
        testing_iters = 127
    elif test == 6:
        w_img, h_img = [624, 352]
        sequence_name = 'Singer2'
        training_iters = 121
        testing_iters = 366
    elif test == 7:
        w_img, h_img = [352, 288]
        sequence_name = 'Woman'
        training_iters = 198
        testing_iters = 597
    elif test == 8:
        w_img, h_img = [640, 480]
        sequence_name = 'David3'
        training_iters = 83
        testing_iters = 252
    elif test == 9:
        w_img, h_img = [320, 240]
        sequence_name = 'Human7'
        training_iters = 83
        testing_iters = 250
    elif test == 10:
        w_img, h_img = [720, 400]
        sequence_name = 'Bird1'
        training_iters = 135
        testing_iters = 408
    elif test == 11:
        w_img, h_img = [360, 240]
        sequence_name = 'Car4'
        training_iters = 219
        testing_iters = 659
    elif test == 12:
        w_img, h_img = [320, 240]
        sequence_name = 'CarDark'
        training_iters = 130
        testing_iters = 393
    elif test == 13:
        w_img, h_img = [320, 240]
        sequence_name = 'Couple'
        training_iters = 46
        testing_iters = 140
    elif test == 14:
        w_img, h_img = [400, 224]
        sequence_name = 'Diving'
        training_iters = 71
        testing_iters = 214
    elif test == 15:
        w_img, h_img = [480, 640]
        sequence_name = 'Human3'
        training_iters = 565
        testing_iters = 1698
    elif test == 16:
        w_img, h_img = [480, 640]
        sequence_name = 'Human6'
        training_iters = 263
        testing_iters = 792
    elif test == 17:
        w_img, h_img = [624, 352]
        sequence_name = 'Singer1'
        training_iters = 116
        testing_iters = 351
    elif test == 18:
        w_img, h_img = [384, 288]
        sequence_name = 'Walking2'
        training_iters = 166
        testing_iters = 500
    elif test == 19:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurCar3'
        training_iters = 117
        testing_iters = 356
    elif test == 20:
        w_img, h_img = [640, 480]
        sequence_name = 'Girl2'
        training_iters = 499
        testing_iters = 1500
    elif test == 21:
        w_img, h_img = [640, 360]
        sequence_name = 'Skating1'
        training_iters = 133
        testing_iters = 400
    elif test == 22:
        w_img, h_img = [320, 240]
        sequence_name = 'Skater'
        training_iters = 50
        testing_iters = 160
    elif test == 23:
        w_img, h_img = [320, 262]
        sequence_name = 'Skater2'
        training_iters = 144
        testing_iters = 435
    elif test == 24:
        w_img, h_img = [320, 246]
        sequence_name = 'Dancer'
        training_iters = 74
        testing_iters = 225
    elif test == 25:
        w_img, h_img = [320, 262]
        sequence_name = 'Dancer2'
        training_iters = 49
        testing_iters = 150
    elif test == 26:
        w_img, h_img = [640, 272]
        sequence_name = 'CarScale'
        training_iters = 81
        testing_iters = 252
    elif test == 27:
        w_img, h_img = [426, 234]
        sequence_name = 'Gym'
        training_iters = 255
        testing_iters = 767
    elif test == 28:
        w_img, h_img = [320, 240]
        sequence_name = 'Human8'
        training_iters = 42
        testing_iters = 128
    elif test == 29:
        w_img, h_img = [416, 234]
        sequence_name = 'Jump'
        training_iters = 40
        testing_iters = 122


    # For MOT 2016:
    # training
    elif test == 30:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-02'
        training_iters = 199
        testing_iters = 600
    elif test == 31:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-04'
        training_iters = 349
        testing_iters = 1050
    elif test == 32:
        w_img, h_img = [640, 480]
        sequence_name = 'MOT16-05'
        training_iters = 278
        testing_iters = 837
    elif test == 33:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-09'
        training_iters = 174
        testing_iters = 525
    elif test == 34:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-10'
        training_iters = 217
        testing_iters = 654
    elif test == 35:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-11'
        training_iters = 299
        testing_iters = 900
    elif test == 36:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-13'
        training_iters = 249
        testing_iters = 750
    # testing
    elif test == 37:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-01'
        training_iters = 149
        testing_iters = 450
    elif test == 38:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-03'
        training_iters = 499
        testing_iters = 1500
    elif test == 39:
        w_img, h_img = [640, 480]
        sequence_name = 'MOT16-06'
        training_iters = 397
        testing_iters = 1194
    elif test == 40:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-07'
        training_iters = 166
        testing_iters = 500
    elif test == 41:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-08'
        training_iters = 208
        testing_iters = 625
    elif test == 42:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-12'
        training_iters = 299
        testing_iters = 900
    elif test == 43:
        w_img, h_img = [1920, 1080]
        sequence_name = 'MOT16-14'
        training_iters = 249
        testing_iters = 750

    # For performance test only
    elif test == 90:
        w_img, h_img = [352, 288]
        sequence_name = 'Jogging_1'
        training_iters = 100
        testing_iters = 300
    elif test == 91:
        w_img, h_img = [352, 288]
        sequence_name = 'Jogging_2'
        training_iters = 100
        testing_iters = 300
    elif test == 92:
        w_img, h_img = [640, 480]
        sequence_name = 'Boy'
        training_iters = 199
        testing_iters = 602
    elif test == 93:
        w_img, h_img = [352, 288]
        sequence_name = 'Jumping'
        training_iters = 103
        testing_iters = 313
    elif test == 94:
        w_img, h_img = [480, 360]
        sequence_name = 'Surfer'
        training_iters = 125
        testing_iters = 376
    elif test == 95:
        w_img, h_img = [640, 332]
        sequence_name = 'Trans'
        training_iters = 41
        testing_iters = 124
    elif test == 96:
        w_img, h_img = [640, 360]
        sequence_name = 'DragonBaby'
        training_iters = 37
        testing_iters = 113
    elif test == 97:
        w_img, h_img = [640, 480]
        sequence_name = 'Liquor'
        training_iters = 580
        testing_iters = 1741
    return [w_img, h_img, sequence_name, training_iters, testing_iters]


def load_yolo_output_test(fold, batch_size, num_steps, id):
        paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        st= id
        ed= id + batch_size*num_steps
        paths_batch = paths[st:ed]

        yolo_output_batch= []
        ct= 0
        for path in paths_batch:
                ct += 1
                yolo_output = np.load(path)
                yolo_output= np.reshape(yolo_output, 4102)
                yolo_output_batch.append(yolo_output)
        yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
        return yolo_output_batch

def save_yolo_output_kalman( out_fold, yolo_output, step, num_steps, batch_size):
        assert(len(yolo_output)== batch_size)
        st= step - 2 #* batch_size * num_steps
        for i in range(batch_size):
            id = st + (i + 1)* num_steps + 1
            pred = yolo_output[i]
            path = os.path.join(out_fold, str(id)+'.txt')
            #np.save(path, pred)
            f=  open(path, 'w')
            f.write(str(pred[0])+ ' '+ str(pred[1])+ ' '+ str(pred[2])+ ' '+ str(pred[3]))


def createFolder( path):
    if not os.path.exists(path):
        os.makedirs(path)


def locations_to_yolo_format(img_wid, img_ht, locations):
    #print("location in func: ", locations[0][0])
    img_wid *= 1.0
    img_ht *= 1.0
    for i in range(len(locations)):
        wid= locations[i][2]-locations[i][0]
        ht= locations[i][3]-locations[i][1]

        # convert top-left point (x,y) to mid point (x, y)
        locations[i][0] += wid/ 2.0
        locations[i][1] += ht/ 2.0

        # convert bottom-right point(x,y) to (wid, ht)
        locations[i][2] = wid
        locations[i][3] = ht

        # convert to [0, 1]
        locations[i][0] /= img_wid
        locations[i][1] /= img_ht
        locations[i][2] /= img_wid
        locations[i][3] /= img_ht
    return locations
#-------------------------------------------------------------
if __name__ == '__main__':

  batch_size = 1
  num_steps = 3

  args = parse_args()
  display = args.display

  display = False # False

  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32,3) #used only for display

  if(display):
    plt.ion()
    fig = plt.figure() 
  
  for test in range(0, 30):
    print('working on sequence: ', test)
    mot_tracker = Sort() #create instance of the SORT tracker

    [w_img, h_img, sequence_name, dummy_1, testing_iters] = choose_video_sequence(test)
    fold_heat = os.path.join('benchmark/DATA/', sequence_name, 'yolo_out')
    output_path = os.path.join('benchmark/DATA/', sequence_name, 'yolo_output_kalman_txt/')
    createFolder(output_path)

    st= 0
    if (sequence_name is 'BlurCar1'): st= 247
    if (sequence_name is 'BlurCar3'): st= 3
    if (sequence_name is 'BlurCar4'): st= 18

    for frame in range(st, testing_iters - 6):
      frame += 1 #detection and frame numbers begin at 1
      seq_dets =  load_yolo_output_test(fold_heat, batch_size, num_steps, frame)

      # Choose the many 4 location parameters and the condfidence parameter that come from each frame
      dets = np.zeros([1, 4])
      dets[0, :] = seq_dets[1][4097:4101] #seq_dets[1][0:4]

      # Convert to [x0,y0,w,h] to [x1,y1,x2,y2]
      #dets[:,2:4] += dets[:,0:2]
      dets[0, 0] *= w_img
      dets[0, 2] *= w_img
      dets[0, 1] *= h_img
      dets[0, 3] *= h_img
      w_half= dets[0, 2]/2.0
      h_half= dets[0, 3]/2.0
      x_mid= dets[0, 0]
      y_mid= dets[0, 1]
      dets[0, 0] = x_mid - w_half
      dets[0, 2] = x_mid + w_half
      dets[0, 1] = y_mid - h_half
      dets[0, 3] = y_mid + h_half

      total_frames += 1

      if(display):
        ax1 = fig.add_subplot(111, aspect='equal')
        fn = 'benchmark/DATA/%s/img/%04d.jpg'%(sequence_name,frame)
        im =io.imread(fn)
        ax1.imshow(im)
        plt.title(sequence_name+' Tracked Targets')

      # Caculate the time spent on tracking
      start_time = time.time()
      trackers = mot_tracker.update(dets)   # detections to tracking results
      cycle_time = time.time() - start_time
      total_time += cycle_time

      if(len(trackers)==0): trackers= [[0, 0, 0, 0, 0]]
      d= trackers[0]
      for i in range(0, 4):
          if np.isnan(d[i]):
              d[i] = 0

      pred_location= [[int(d[0]), int(d[1]), int(d[2]-d[0]),int(d[3]-d[1])]]   # (X1, Y1, W, H)
      save_yolo_output_kalman(output_path, pred_location, frame, num_steps, batch_size)

      for d in trackers:
        #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
        if(display):
          ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
          ax1.set_adjustable('box-forced')

      if(display):
        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  if(display):
    print("Note: to get real runtime results run without the option: --display")
  


