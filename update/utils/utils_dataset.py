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
Script File:
    ROLO_utils.py
    [Input] A network model, a file
    [Output] A file with Detection or Tracking results
Description:
	ROLO is short for Recurrent YOLO, aimed for object detection, tracking and predicting
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

import cv2
import os, sys, time, math, re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_io_coord import load_lines_from_txt_file, load_regular_coord_by_line
from utils_convert_coord import coord_regular_to_decimal

def batchload_gt_decimal_coords_from_VID(VID_annotation_path, batch_seq_names, batch_frame_ids, offset = 3):
    batch_decimal_coords = []
    batch_seq_paths = [os.path.join(VID_annotation_path, seq_name)
                      for seq_name in batch_seq_names]

    for id, seq_path in enumerate(batch_seq_paths):
        frame_id = batch_frame_ids[id]
        line_id = frame_id + offset # Prediction of future frame

        info_file_path = find_sequence_info_file_from_VID(seq_path)
        [img_wid, img_ht] = load_sequence_info(info_file_path)

        gt_file_path = find_sequence_gt_file_from_VID(seq_path)
        decimal_coord = load_gt_decimal_coords_from_file(gt_file_path, line_id, img_wid, img_ht)
        batch_decimal_coords.append(decimal_coord)

    return batch_decimal_coords


def load_gt_decimal_coords_from_file(gt_file_path, line_id, img_wid, img_ht):
    lines = load_lines_from_txt_file(gt_file_path)
    regular_coord = load_regular_coord_by_line(lines, line_id)
    if regular_coord is False: return False

    decimal_coord = coord_regular_to_decimal(regular_coord, img_wid, img_ht)
    return decimal_coord


def find_sequence_info_file_from_VID(seq_path):
    info_file_path = os.path.join(seq_path, "sequence_info.txt")
    return info_file_path


def find_sequence_gt_file_from_VID(seq_path):
    gt_file_path = os.path.join(seq_path, "groundtruth_rect.txt")
    return gt_file_path


def load_sequence_info(info_file_path):
    with open(info_file_path, "r") as text_file:
        lines = text_file.read().split(' ')
        [img_wid, img_ht, sequence_name, training_iters] = [int(lines[0]), int(lines[1]), lines[2], int(lines[3])]
    return  [img_wid, img_ht]


def choose_video_sequence_from_VID_by_id(folder, i):
    if i< 1000:
        mfolder = folder + '/ILSVRC2015_VID_train_0000'
        j = i
    elif i < 2000:
        mfolder = folder + '/ILSVRC2015_VID_train_0001'
        j = i%1000
    elif i < 3000:
        mfolder = folder + '/ILSVRC2015_VID_train_0002'
        j = i%2000
    else:
        mfolder = folder + '/ILSVRC2015_VID_train_0003'
        j = i%3000
    subfolders = get_immediate_subfolder_names(mfolder)
    subfolder_sequence_info_file = os.path.join(mfolder, subfolders[j], 'sequence_info.txt')
    with open(subfolder_sequence_info_file, "r") as text_file:
        lines = text_file.read().split(' ')
        [img_wid, img_ht, sequence_name, training_iters] = [int(lines[0]), int(lines[1]), lines[2], int(lines[3])]
    return  [img_wid, img_ht, sequence_name, training_iters]


def choose_video_sequence_from_OTB50(test):
    start_frame= 1
    # For VOT-50:
    if test == 0:
        w_img, h_img = [576, 432]
        sequence_name = 'Basketball'
        testing_iters = 725
    elif test == 1:
        w_img, h_img = [640, 360]
        sequence_name = 'Biker'
        testing_iters = 142
    elif test == 2:
        w_img, h_img = [720, 400]
        sequence_name = 'Bird1'
        testing_iters = 408
    elif test == 3:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurBody'
        testing_iters = 334
    elif test == 4:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurCar2'
        testing_iters = 585
    elif test == 5:
        w_img, h_img = [640, 480]    #
        sequence_name = 'BlurFace'
        testing_iters = 493
    elif test == 6:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurOwl'
        testing_iters = 631
    elif test == 7:
        w_img, h_img = [640, 360]
        sequence_name = 'Bolt'
        testing_iters = 350
    elif test == 8:
        w_img, h_img = [640, 480]
        sequence_name = 'Box'
        testing_iters = 1161
    elif test == 9:
        w_img, h_img = [320, 240]
        sequence_name = 'Car1'
        testing_iters = 1020
    elif test == 10:
        w_img, h_img = [360, 240]
        sequence_name = 'Car4'
        testing_iters = 659
    elif test == 11:
        w_img, h_img = [320, 240]
        sequence_name = 'CarDark'
        testing_iters = 393
    elif test == 12:
        w_img, h_img = [640, 272]
        sequence_name = 'CarScale'
        testing_iters = 252
    elif test == 13:
        w_img, h_img = [320, 240]
        sequence_name = 'ClifBar'
        testing_iters = 472
    elif test == 14:
        w_img, h_img = [320, 240]
        sequence_name = 'Couple'
        testing_iters = 140
    elif test == 15:
        w_img, h_img = [600, 480]
        sequence_name = 'Crowds'
        testing_iters = 347
    elif test == 16:
        w_img, h_img = [320, 240]   #
        sequence_name = 'David'
        testing_iters = 770
        start_frame= 300
    elif test == 17:
        w_img, h_img = [704, 400]
        sequence_name = 'Deer'
        testing_iters = 71
    elif test == 18:
        w_img, h_img = [400, 224]
        sequence_name = 'Diving'
        testing_iters = 214
    elif test == 19:
        w_img, h_img = [640, 360]
        sequence_name = 'DragonBaby'
        testing_iters = 113
    elif test == 20:
        w_img, h_img = [720, 480]   #
        sequence_name = 'Dudek'
        testing_iters = 1145
    elif test == 21:
        w_img, h_img = [624, 352]    #
        sequence_name = 'Football'
        testing_iters = 74
    elif test == 22:
        w_img, h_img = [360, 240]     #
        sequence_name = 'Freeman4'
        testing_iters = 283
    elif test == 23:
        w_img, h_img = [128, 96]     #
        sequence_name = 'Girl'
        testing_iters = 500
    elif test == 24:
        w_img, h_img = [480, 640]
        sequence_name = 'Human3'
        testing_iters = 1698
    elif test == 25:
        w_img, h_img = [640, 480]
        sequence_name = 'Human4'
        testing_iters = 667
    elif test == 26:
        w_img, h_img = [480, 640]
        sequence_name = 'Human6'
        testing_iters = 792
    elif test == 27:
        w_img, h_img = [320, 240]
        sequence_name = 'Human9'
        testing_iters = 302
    elif test == 28:
        w_img, h_img = [720, 304]   #
        sequence_name = 'Ironman'
        testing_iters = 166
    elif test == 29:
        w_img, h_img = [416, 234]
        sequence_name = 'Jump'
        testing_iters = 122
    elif test == 30:
        w_img, h_img = [352, 288]   #
        sequence_name = 'Jumping'
        testing_iters = 313
    elif test == 31:
        w_img, h_img = [640, 480]
        sequence_name = 'Liquor'
        testing_iters = 1741
    elif test == 32:
        w_img, h_img = [800, 336]    #
        sequence_name = 'Matrix'
        testing_iters = 100
    elif test == 33:
        w_img, h_img = [640, 360]
        sequence_name = 'MotorRolling'
        testing_iters = 164
    elif test == 34:
        w_img, h_img = [312, 233]
        sequence_name = 'Panda'
        testing_iters = 1000
    elif test == 35:
        w_img, h_img = [352, 240]
        sequence_name = 'RedTeam'
        testing_iters = 1918
    elif test == 36:
        w_img, h_img = [624, 352]   #
        sequence_name = 'Shaking'
        testing_iters = 365
    elif test == 37:
        w_img, h_img = [624, 352]
        sequence_name = 'Singer2'
        testing_iters = 366
    elif test == 38:
        w_img, h_img = [640, 360]
        sequence_name = 'Skating1'
        testing_iters = 400
    elif test == 39:
        w_img, h_img = [640, 352]
        sequence_name = 'Skating2-1'
        testing_iters = 473
    elif test == 40:
        w_img, h_img = [640, 352]
        sequence_name = 'Skating2-2'
        testing_iters = 473
    elif test == 41:
        w_img, h_img = [640, 360]
        sequence_name = 'Skiing'
        testing_iters = 81
    elif test == 42:
        w_img, h_img = [640, 360]   #
        sequence_name = 'Soccer'
        testing_iters = 392
    elif test == 43:
        w_img, h_img = [480, 360]
        sequence_name = 'Surfer'
        testing_iters = 376
    elif test == 44:
        w_img, h_img = [320, 240]
        sequence_name = 'Sylvester'
        testing_iters = 1345
    elif test == 45:
        w_img, h_img = [640, 480]
        sequence_name = 'Tiger2'
        testing_iters = 365
    elif test == 46:
        w_img, h_img = [320, 240]   #
        sequence_name = 'Trellis'
        testing_iters = 569
    elif test == 47:
        w_img, h_img = [768, 576]
        sequence_name = 'Walking'
        testing_iters = 412
    elif test == 48:
        w_img, h_img = [384, 288]
        sequence_name = 'Walking2'
        testing_iters = 500
    elif test == 49:
        w_img, h_img = [352, 288]
        sequence_name = 'Woman'
        testing_iters = 597

    # For VOT-2015, read the list.txt and get the corresponding sequences.


    return [w_img, h_img, sequence_name, start_frame, testing_iters]
