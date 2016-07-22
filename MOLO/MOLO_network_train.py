# Copyright (c) <2016> <GUANGHAN NING>. All Rights Reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

'''
Script File: MOLO_network_train.py

Description:
	MOLO is short for Multi-target ROLO, aimed at simultaneous detection and tracking of multiple targets
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

# Imports
import ROLO_utils as utils

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import cv2

import numpy as np
import os.path
import time
import random


class ROLO_TF:
    disp_console = False
    restore_weights = True

    # YOLO parameters
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    yolo_weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    w_img, h_img = [352, 240]

    # ROLO Network Parameters
    rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/output/MOLO/model_MOT.ckpt'
    lstm_depth = 3
    num_steps = 3  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict # data input: 4096+6= 5002

    # ROLO Training Parameters
    #learning_rate = 0.00001 #training
    learning_rate = 0.00001 #testing

    training_iters = 210#100000
    batch_size = 1 #128
    display_step = 1

    # tf Graph input
    x = tf.placeholder("float32", [None, num_steps, num_input])
    istate = tf.placeholder("float32", [None, 2*num_input]) #state & cell => 2x num_input
    y = tf.placeholder("float32", [None, num_gt])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }


    def __init__(self,argvs = []):
        print("ROLO init")
        self.ROLO(argvs)


    def LSTM_single(self, name,  _X, _istate, _weights, _biases):
        with tf.device('/gpu:0'):
            # input shape: (batch_size, n_steps, n_input)
            _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
            # Reshape to prepare input to hidden activation
            _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _X = tf.split(0, self.num_steps, _X) # n_steps * (batch_size, num_input)
            #print("_X: ", _X)

        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
        state = _istate
        for step in range(self.num_steps):
            outputs, state = tf.nn.rnn(cell, [_X[step]], state)
            tf.get_variable_scope().reuse_variables()

        #print("output: ", outputs)
        #print("state: ", state)
        return outputs


    # Experiment with dropout
    def dropout_features(self, feature, prob):
        if prob == 0: return feature
        else:
            num_drop = int(prob * 4096)
            drop_index = random.sample(xrange(4096), num_drop)
            for i in range(len(drop_index)):
                index = drop_index[i]
                feature[index] = 0
            return feature


    # Experiment with input box noise (translate, scale)
    def det_add_noise(self, det):
        translate_rate = random.uniform(0.98, 1.02)
        scale_rate = random.uniform(0.8, 1.2)

        det[0] *= translate_rate
        det[1] *= translate_rate
        det[2] *= scale_rate
        det[3]*= scale_rate

        return det


    '''---------------------------------------------------------------------------------------'''
    def build_networks(self):
        if self.disp_console : print "Building MOLO graph..."

        # Build rolo layers
        self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.ious= tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console : print "Loading complete!" + '\n'


    def train_7(self):
        print("TRAINING MOLO...")
        log_file = open("output/trainging-7-log.txt", "a") #open in append mode
        self.build_networks()

        ''' TUNE THIS'''
        offset = 30
        num_videos = 7
        epoches = 7 * 300


        # Use rolo_input for LSTM training
        pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        self.pred_location = pred[0][:, 4097:4101]
        self.correct_prediction = tf.square(self.pred_location - self.y)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        self.learning_rate = 0.00001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy) # Adam Optimizer

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            avg_loss = 0
            stay_epoch = 2
            for epoch in range(2, epoches):
                #if(avg_loss > 0.1):
                #    epoch= stay_epoch
                #print(epoch)
                i = epoch % num_videos + offset
                [self.w_img, self.h_img, sequence_name, self.training_iters, self.testing_iters]= utils.choose_video_sequence(i)

                x_path = os.path.join('benchmark/MOT/MOT2016/train/', sequence_name, 'yolo_out/')
                seq_dets = np.loadtxt('3rd_party/sort-master/output/%s.txt'%(sequence_name),delimiter=',') #load detections

                y_path = os.path.join('benchmark/MOT/MOT2016/train/', sequence_name, 'gt/gt.txt')

                #out_file = open('output/MOLO/%s.txt'%(sequence_name),'w')

                #self.output_path = os.path.join('benchmark/MOT/MOT2016/train/', sequence_name, 'molo_out_train/')
                #utils.createFolder(self.output_path)
                total_loss = 0
                id = 1
                box_num= 0

                # Keep training until reach max iterations
                while id  < self.testing_iters- self.num_steps: # + 1
                    # Load training data & ground truth
                    batch_xs_raw = self.rolo_utils.load_yolo_feat_test_MOLO(x_path, self.batch_size, self.num_steps, id-1)  # 3 features: (id-1, id, id+1), start from 0.
                    dets_last = seq_dets[ (seq_dets[:,0]== (id))&(seq_dets[:,6]==1) , 1:6]  # dets starts from 1
                    dets = seq_dets[ (seq_dets[:,0]== (id+1))&(seq_dets[:,6]==1) , 1:6]
                    dets_next = seq_dets[ (seq_dets[:,0]== (id+2))&(seq_dets[:,6]==1) , 1:6]

                    for person in range(len(dets)):
                        box_num += 1

                        batch_ys = [dets[person][1:5]]
                        batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                        # Merge the features with dets in batch_xs
                        person_id = dets[person][0]            #BUG!!!!:    (-1 is wrong)   -1  # person_id starts from 1, but index starts from 0, so minus 1
                        loc_last = dets_last[dets_last[:,0]==person_id, 1:5]
                        loc_prst = dets[dets[:,0]==person_id, 1:5]
                        loc_next = dets_next[dets_next[:,0]==person_id, 1:5]
                        #print('loca_last', loc_last[0][:])
                        #print('batch_xs', batch_xs_raw[0])
                        if len(loc_last) == 0 or len(loc_next)==0:
                            continue
                        loc_last = utils.locations_from_0_to_1(self.w_img, self.h_img, [loc_last[0][:]])
                        loc_prst = utils.locations_from_0_to_1(self.w_img, self.h_img, [loc_prst[0][:]])
                        loc_next = utils.locations_from_0_to_1(self.w_img, self.h_img, [loc_next[0][:]])

                        # Add noise to batch_xs
                        #loc_last[0]= self.det_add_noise( loc_last[0])
                        #loc_prst[0]= self.det_add_noise( loc_prst[0])
                        #loc_next[0]= self.det_add_noise( loc_next[0])

                        #print('loca_last', loc_last[0][:])
                        batch_xs_raw[0][4097:4101] = loc_last[0][:]
                        batch_xs_raw[1][4097:4101] = loc_prst[0][:]
                        batch_xs_raw[2][4097:4101] = loc_next[0][:]

                        # Reshape data to get 3 seq of 5002 elements
                        batch_xs = np.reshape(batch_xs_raw, [self.batch_size, self.num_steps, self.num_input])
                        batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                        #print("Batch_ys: ", batch_ys)

                        # Output prediction to txt file
                        pred_location= sess.run(self.pred_location,feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                        d = utils.locations_normal(self.w_img, self.h_img, pred_location[0])  # d = [x_mid, y_mid, w, h] in pixels
                        #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(id+1, person_id, d[0]-d[2]/2.0, d[1]- d[3]/2.0, d[2], d[3]), out_file)
                        #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(id+1, person_id, d[0]-d[2]/2.0, d[1]- d[3]/2.0, d[2], d[3]))

                        if self.disp_console: print("ROLO Pred: ", pred_location)
                        if self.disp_console: print("ROLO Pred in pixel: ", pred_location[0][0]*self.w_img, pred_location[0][1]*self.h_img, pred_location[0][2]*self.w_img, pred_location[0][3]*self.h_img)

                        sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})

                        # Calculate batch loss
                        loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                        if self.disp_console: print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) #+ "{:.5f}".format(self.accuracy)
                        total_loss += loss
                    id += 1

                #out_file.close()

                #print "Optimization Finished!"
                avg_loss = total_loss/box_num
                print "Avg loss: " + sequence_name + ": " + str(avg_loss)

                log_file.write(str("{:.3f}".format(avg_loss)) + '  ')
                if epoch % num_videos == 0:
                    log_file.write('\n')
                save_path = self.saver.save(sess, self.rolo_weights_file)
                print("Model saved in file: %s" % save_path)

        log_file.close()
        return


    def ROLO(self, argvs):

            self.rolo_utils= utils.ROLO_utils()
            self.rolo_utils.loadCfg()
            self.params = self.rolo_utils.params

            arguments = self.rolo_utils.argv_parser(argvs)

            if self.rolo_utils.flag_train is True:
                self.training(utils.x_path, utils.y_path)
            elif self.rolo_utils.flag_track is True:
                self.build_networks()
                self.track_from_file(utils.file_in_path)
            elif self.rolo_utils.flag_detect is True:
                self.build_networks()
                self.detect_from_file(utils.file_in_path)
            else:
                self.train_7()

    '''----------------------------------------main-----------------------------------------------------'''
def main(argvs):
        ROLO_TF(argvs)

if __name__=='__main__':
        main(' ')

