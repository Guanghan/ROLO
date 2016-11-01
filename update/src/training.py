import sys, os
sys.path.append(os.path.abspath("../utils/"))
import time, random

from utils_io_coord import *
from utils_io_list import *
from utils_dataset import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rnn, rnn_cell, cv2

from testing import test

class ROLO_TF:
    # Buttons
    validate = True
    validate_step = 1000
    display_validate = True
    save_step = 1000
    display_step = 1
    restore_weights = True
    display_coords = False
    display_regu = False

    # Magic numbers
    learning_rate = 0.0001
    lamda = 1.0

    # Path
    list_pairs_numpy_file_path = '/home/ngh/dev/ROLO-TRACK/training_list/list_0.npy'
    dataset_annotation_folder_path = '/home/ngh/dev/ROLO-dev/benchmark/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000'
    numpy_folder_name = 'VID_loc_gt'   # Alternatives: 'VID_loc_gt' and 'VID_loc'
    rolo_weights_file = '../rolo_weights.ckpt'
    rolo_current_save = '../rolo_weights_temp.ckpt'

    # Vector
    len_feat = 4096
    len_predict = 6
    len_coord = 4
    len_vec = 4102

    # Batch
    nsteps = 3
    batchsize = 16
    n_iters = 180000
    batch_offset = 0

    # Data
    x = tf.placeholder("float32", [None, nsteps, len_vec])
    y = tf.placeholder("float32", [None, len_coord])
    istate = tf.placeholder("float32", [None, 2*len_vec])
    list_batch_pairs = []

    # Initializing
    def __init__(self, argvs = []):
        print("ROLO Initializing")
        self.ROLO()


    # Routines: Data
    def load_training_list(self):
        self.list_batch_pairs = load_list_batch_pairs_from_numpy_file(self.list_pairs_numpy_file_path,
                                                                      self.batchsize)


    def load_batch(self, b_id):
        max_id = len(self.list_batch_pairs)
        if b_id <= max_id:
            batch_pairs = self.list_batch_pairs[b_id]
            batch_frame_ids = [int(batch_pair[1]) for batch_pair in batch_pairs]

            batch_subfolder_names = [batch_pair[0] for batch_pair in batch_pairs]
            batch_numpy_folder_paths = [os.path.join(self.dataset_annotation_folder_path,
                                                     subfolder_name,
                                                     self.numpy_folder_name)
                                        for subfolder_name in batch_subfolder_names]

            attempted_batch_yolovecs = batchload_yolovecs_from_numpy_folders(batch_numpy_folder_paths,
                                                                             batch_frame_ids,
                                                                             self.batchsize,
                                                                             self.nsteps)
        if b_id > max_id or attempted_batch_yolovecs == -1:
            self.update_dataset_annotation_folder_path()
            self.batch_offset = self.iter_id
            self.load_training_list()
            attempted_batch_yolovecs = False
            batch_subfolder_names = []
            batch_frame_ids = []
        return [attempted_batch_yolovecs, batch_subfolder_names, batch_frame_ids]


    def update_dataset_annotation_folder_path(self):
        try:
            list_folder_path = list(self.dataset_annotation_folder_path)
            list_file_path = list(self.list_pairs_numpy_file_path)

            last_int = int(self.dataset_annotation_folder_path[-1])
            new_int = (last_int + 1)%4
            list_folder_path[-1] = str(new_int)
            list_file_path[-5] = str(new_int)

            self.dataset_annotation_folder_path = ''.join(list_folder_path)
            self.list_pairs_numpy_file_path = ''.join(list_file_path)
            print(self.dataset_annotation_folder_path)
            print(self.list_pairs_numpy_file_path)
            print("Finished 1/4 of all data. Annotation folder updated")
        except ValueError:
            print("Error updating dataset annotation folder")


    # Routines: Network
    def LSTM(self, name,  _X, _istate):
        ''' shape: (batchsize, nsteps, len_vec) '''
        _X = tf.transpose(_X, [1, 0, 2])
        ''' shape: (nsteps, batchsize, len_vec) '''
        _X = tf.reshape(_X, [self.nsteps * self.batchsize, self.len_vec])
        ''' shape: n_steps * (batchsize, len_vec) '''
        _X = tf.split(0, self.nsteps, _X)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.len_vec, self.len_vec, state_is_tuple = False)
        state = _istate
        for step in range(self.nsteps):
            pred, state = rnn.rnn(lstm_cell, [_X[step]], state, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            if step == 0:   output_state = state

        batch_pred_feats = pred[0][:, 0:4096]
        batch_pred_coords = pred[0][:, 4097:4101]
        return batch_pred_feats, batch_pred_coords, output_state


    # Routines: Train & Test
    def train(self):
        ''' Network '''
        batch_pred_feats, batch_pred_coords, self.final_state = self.LSTM('lstm', self.x, self.istate)

        ''' Loss: L2 '''
        loss = tf.reduce_mean(tf.square(self.y - batch_pred_coords)) * 100

        ''' regularization term: L2 '''
        regularization_term = tf.reduce_mean(tf.square(self.x[:, self.nsteps-1, 0:4096] - batch_pred_feats)) * 100

        ''' Optimizer '''
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss  + self.lamda * regularization_term) # Adam Optimizer

        ''' Summary for tensorboard analysis '''
        dataset_loss = -1
        dataset_loss_best = 100
        test_writer = tf.train.SummaryWriter('summary/test')
        tf.scalar_summary('dataset_loss', dataset_loss)
        summary_op = tf.merge_all_summaries()

        ''' Initializing the variables '''
        init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        batch_states = np.zeros((self.batchsize, 2*self.len_vec))

        ''' Launch the graph '''
        with tf.Session() as sess:
            if self.restore_weights == True and os.path.isfile(self.rolo_current_save):
                sess.run(init)
                self.saver.restore(sess, self.rolo_current_save)
                print("Weight loaded, finetuning")
            else:
                sess.run(init)
                print("Training from scratch")

            self.load_training_list()

            for self.iter_id in range(self.n_iters):
                ''' Load training data & ground truth '''
                batch_id = self.iter_id - self.batch_offset
                [batch_vecs, batch_seq_names, batch_frame_ids] = self.load_batch(batch_id)
                if batch_vecs is False: continue

                batch_xs = batch_vecs
                batch_ys = batchload_gt_decimal_coords_from_VID(self.dataset_annotation_folder_path,
                                                                batch_seq_names,
                                                                batch_frame_ids,
                                                                offset = self.nsteps - 1)
                if batch_ys is False: continue

                ''' Reshape data '''
                batch_xs = np.reshape(batch_xs, [self.batchsize, self.nsteps, self.len_vec])
                batch_ys = np.reshape(batch_ys, [self.batchsize, 4])

                ''' Update weights by back-propagation '''
                sess.run(optimizer, feed_dict={self.x: batch_xs,
                                               self.y: batch_ys,
                                               self.istate: batch_states})

                if self.iter_id % self.display_step == 0:
                    ''' Calculate batch loss '''
                    batch_loss = sess.run(loss,
                                          feed_dict={self.x: batch_xs,
                                                     self.y: batch_ys,
                                                     self.istate: batch_states})
                    print("Batch loss for iteration %d: %.3f" % (self.iter_id, batch_loss))

                if self.display_regu is True:
                    ''' Caculate regularization term'''
                    batch_regularization = sess.run(regularization_term,
                                                    feed_dict={self.x: batch_xs,
                                                               self.y: batch_ys,
                                                               self.istate: batch_states})
                    print("Batch regu for iteration %d: %.3f" % (self.iter_id, batch_regularization))

                if self.display_coords is True:
                    ''' Caculate predicted coordinates '''
                    coords_predict = sess.run(batch_pred_coords,
                                              feed_dict={self.x: batch_xs,
                                                         self.y: batch_ys,
                                                         self.istate: batch_states})
                    print("predicted coords:" + str(coords_predict[0]))
                    print("ground truth coords:" + str(batch_ys[0]))

                ''' Save model '''
                if self.iter_id % self.save_step == 1:
                    self.saver.save(sess, self.rolo_current_save)
                    print("\n Model saved in file: %s" % self.rolo_current_save)

                ''' Validation '''
                if self.validate == True and self.iter_id % self.validate_step == 0:
                    dataset_loss = test(self, sess, loss, batch_pred_coords)

                    ''' Early-stop regularization '''
                    if dataset_loss <= dataset_loss_best:
                        dataset_loss_best = dataset_loss
                        self.saver.save(sess, self.rolo_weights_file)
                        print("\n Better Model saved in file: %s" % self.rolo_weights_file)

                    ''' Write summary for tensorboard '''
                    summary = sess.run(summary_op, feed_dict={self.x: batch_xs,
                                                              self.y: batch_ys,
                                                              self.istate: batch_states})
                    test_writer.add_summary(summary, self.iter_id)
        return


    def ROLO(self):
        print("Initializing ROLO")
        self.train()
        print("Training Completed")

'''----------------------------------------main-----------------------------------------------------'''
def main(argvs):
    ROLO_TF(argvs)

if __name__ == "__main__":
    main(' ')
