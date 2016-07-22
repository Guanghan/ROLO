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
Script File: ROLO_utils.py

Description:
      
	ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''
import os, sys, time
import numpy as np
import ConfigParser
import cv2
import pickle
import tensorflow as tf
import math
global Config
import matplotlib.pyplot as plt

class ROLO_utils:
        cfgPath = ''
        Config = []
        flag_train = False
        flag_track = False
        flag_detect = False
        params= {
        'img_size': [448, 448],
        'num_classes': 20,
        'alpha': 0.1,
        'dropout_rate': 0.5,
        'num_feat_lstm': 5001,
        'num_steps': 28,
        'batch_size': 1,
        'tensor_size': 7,
        'predict_num_per_tensor': 2,
        'threshold': 0.2,
        'iou_threshold': 0.5,
        'class_labels': ["stop", "vehicle", "pedestrian"],

        'conv_layer_num': 3,
        'conv_filters': [16, 32, 64],
        'conv_size': [3, 3, 3],
        'conv_stride': [1,1,1],

        'fc_layer_num': 3,
        'fc_input': [1024, 1024, 1024],
        'fc_output': [1024, 1024, 1024]
        }
        file_weights= None
        file_in_path= None
        file_out_path= None
        flag_write= False
        flag_show_img= False

        batch_size = 128

        x_path= "u03/yolo_output/"
        y_path= "u03/rolo_gt"

        def __init__(self,argvs = []):
            print("Utils init")

        # Network Parameters
        def loadCfg(self):
                Config = ConfigParser.ConfigParser()
                Config.read(self.cfgPath)
                Sections = Config.sections()

                print('self.cfgPath=' + self.cfgPath)
                if os.path.isfile(self.cfgPath):
                        dict_parameters = self.ConfigSectionMap(Config, "Parameters")
                        dict_networks = self.ConfigSectionMap(Config, "Networks")

                        self.params['img_size']= dict_parameters['img_size']   #[448, 448]
                        self.params['alpha'] = dict_parameters['alpha']
                        self.params['num_classes']=  dict_parameters['num_classes']   #20
                        self.params['dropout_rate']=  dict_parameters['dropout_rate'] # 0.5
                        self.params['num_feat_lstm']=  dict_parameters['num_feat_lstm'] #4096+5 # number of features in hidden layer of LSTM
                        self.params['num_steps']=  dict_parameters['num_steps'] #  28 # timesteps for LSTM
                        self.params['batch_size']=  dict_parameters['batch_size'] # 1 # during testing it is 1; during training it is 64.
                        self.params['tensor_size'] = dict_parameters['tensor_size']
                        self.params['predict_num_per_tensor'] = dict_parameters['predict_num_per_tensor']
                        self.params['threshold'] = dict_parameters['tensor_size']
                        self.params['iou_threshold'] = dict_parameters['iou_threshold']

                        self.params['conv_layer_num'] = dict_networks['conv_layer_num']
                        self.params['conv_filters']= dict_networks['conv_filters']
                        self.params['conv_size']= dict_networks['conv_size']
                        self.params['conv_stride']= dict_networks['conv_stride']
                        self.params['fc_layer_num']= dict_networks['fc_layer_num']
                        self.params['fc_input'] = dict_networks['fc_input']
                        self.params['fc_output'] = dict_networks['fc_output']

                return self.params


        def ConfigSectionMap(self, Config, section):
                dict1= {}
                options = Config.options(section)
                for option in options:
                        dict1[option] = Config.get(section, option)
                return dict1


        def validate_file_format(self, file_in_path, allowed_format):
                if not os.path.isfile(file_in_path) or os.path.splitext(file_in_path)[1][1:] not in allowed_format:
                        #print(os.path.splitext(file_in_path)[1][1:])
                        print "Input file with correct format not found.\n"
                        return False
                else:
                        return True


        def argv_parser(self, argvs):
                #global  file_weights, file_in_path, file_out_path, flag_write, flag_show_img
                allowed_format = ['png', 'jpg', 'JPEG', 'avi', 'mp4', 'mkv','cfg']
                for i in range(1, len(argvs), 2):
                        if argvs[i] == '-train': self.flag_train= True
                        if argvs[i] == '-cfg':  self.cfgPath = argvs[i+1]
                        if argvs[i] == '-weights': self.file_weights = argvs[i+1]
                        if argvs[i] == '-input':  self.file_in_path = argvs[i+1];  self.validate_file_format(file_in_path, allowed_format)
                        if argvs[i] == '-output': self.file_out_path = argvs[i+1]; self.flag_write = True
                        if argvs[i] == '-detect': self.flag_detect = True; self.flag_track= False;
                        if argvs[i] == '-track': self.flag_detect= True; self.flag_track = True;
                        if argvs[i] == '-imshow':
                                if argvs[i+1] == '1': self.flag_show_img = True
                                else: self.flag_show_img = False
                return (self.cfgPath, self.file_weights, self.file_in_path)


        def is_image(self, file_in_path):
                if  os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in ['jpg', 'JPEG', 'png', 'JPG']:
                        return True
                else:
                        return False


        def is_video(self, file_in_path):
                if  os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in ['avi', 'mkv', 'mp4']:
                        return True
                else:
                        return False


        # Not Face user
        def file_to_img(self, filepath):
            print 'Processing '+ filepath
            img = cv2.imread(filepath)
            return img


        def file_to_video(self, filepath):
            print 'processing '+ filepath
            try:
                    video = cv2.VideoCapture(filepath)
            except IOError:
                    print 'cannot open video file: ' + filepath
            else:
                    print 'unknown error reading video file'
            return video


        def iou(self,box1,box2):
                tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
                lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
                if tb < 0 or lr < 0 : intersection = 0
                else : intersection =  tb*lr
                return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


        def find_iou_cost(self, pred_locs, gts):
                # for each element in the batch, find its iou. output a list of ious.
                cost = 0
                #batch_size= len(pred_locs)
                batch_size= self.batch_size
                #assert (len(gts)== batch_size)
                #print("gts: ", gts)
                #print("batch_size: ", batch_size)
                #print("pred_locs: ", pred_locs)
                #ious = []
                ious = np.zeros((batch_size, 4))

                for i in range(batch_size):
                        pred_loc = pred_locs[i,:]
                        #print("pred_loc[i]: ", pred_loc)
                        gt = gts[i,:]
                        iou_ = self.iou(pred_loc, gt)
                        #ious.append(iou_)
                        #print("iou_", iou_)
                        ious[i,:]= iou_
                #ious= tf.reshape(ious, batch_size)
                #print("ious: ", ious)
                '''
                avg_iou= 0
                for i in range(batch_size):
                        pred_loc = pred_locs[i,:]
                        gt= gts[i,:]
                        print("gt", gt)
                        #print("pred_loc", pred_loc)
                        avg_iou += self.iou(pred_loc, gt)
                avg_iou /= batch_size

                print("avg_iou shape: ", tf.shape(avg_iou)) # single tensor expected
                return avg_iou'''
                return ious


        def load_folder(self, path):
                paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
                return paths

        def load_dataset_gt(self, gt_file):
                txtfile = open(gt_file, "r")
                lines = txtfile.read().split('\n')  #'\r\n'
                return lines

        def find_gt_location(self, lines, id):
                #print("lines length: ", len(lines))
                #print("id: ", id)
                line = lines[id]
                elems = line.split('\t')   # for gt type 2
                #print(elems)
                if len(elems) < 4:
                        elems = line.split(',') #for gt type 1
                        #print(elems)
                x1 = elems[0]
                y1 = elems[1]
                w = elems[2]
                h = elems[3]
                gt_location = [int(x1), int(y1), int(w), int(h)]
                return gt_location


        def find_best_location(self, locations, gt_location):
                # locations (class, x, y, w, h, prob); (x, y) is the middle pt of the rect
                # gt_location (x1, y1, w, h)
                x1 = gt_location[0]
                y1 = gt_location[1]
                w = gt_location[2]
                h = gt_location[3]
                gt_location_revised= [x1 + w/2, y1 + h/2, w, h]

                max_ious= 0
                for location, id in enumerate(locations):
                        location_revised = location[1:5]
                        ious = self.iou(location_revised, gt_location_revised)
                        if ious >= max_ious:
                                max_ious = ious
                                index = id
                return locations[index]


        def save_yolo_output(self, out_fold, yolo_output, filename):
                name_no_ext= os.path.splitext(filename)[0]
                output_name= name_no_ext + ".yolo"
                path = os.path.join(out_fold, output_name)
                pickle.dump(yolo_output, open(path, "rb"))


        def load_yolo_output(self, fold, batch_size, num_steps, step):
                paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
                paths = sorted(paths)
                st= step*batch_size*num_steps
                ed= (step+1)*batch_size*num_steps
                paths_batch = paths[st:ed]

                yolo_output_batch= []
                ct= 0
                for path in paths_batch:
                        ct += 1
                        #yolo_output= pickle.load(open(path, "rb"))
                        yolo_output = np.load(path)
                    
                        yolo_output= np.reshape(yolo_output, 4102)
                        yolo_output[4096]= 0
                        yolo_output[4101]= 0
                        yolo_output_batch.append(yolo_output)
                print(yolo_output_batch)
                yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
                return yolo_output_batch


        def load_rolo_gt(self, path, batch_size, num_steps, step):
                lines= self.load_dataset_gt(path)
                offset= num_steps - 2  # offset is for prediction of the future
                st= step*batch_size*num_steps
                ed= (step+1)*batch_size*num_steps
                #print("st: " + str(st))
                #print("ed: " + str(ed))
                batch_locations= []
                for id in range(st+offset, ed+offset, num_steps):
                        location= self.find_gt_location(lines, id)
                        batch_locations.append(location)
                return batch_locations


        def load_yolo_output_test(self, fold, batch_size, num_steps, id):
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
                        #print(yolo_output)
                        yolo_output= np.reshape(yolo_output, 4102)
                        yolo_output_batch.append(yolo_output)
                yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
                return yolo_output_batch





        def load_yolo_feat_test_MOLO(self, fold, batch_size, num_steps, id):
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
                        #print(yolo_output[0][0][0])
                        #print(len(yolo_output[0][0][0]))
                        yolo_output_new=  np.concatenate(
				                  ( np.reshape(yolo_output[0][0][0], [-1, 4096]),
								    np.reshape([0,0,0,0,0,0], [-1, 6]) ),
								  axis = 1)
                        yolo_output_new= np.reshape(yolo_output_new, 4102)
                        yolo_output_batch.append(yolo_output_new)
                yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
                #print 'yolo_output_batch:' + str(yolo_output_batch)
                return yolo_output_batch


        def load_yolo_output_test_MOLO(self, fold, batch_size, num_steps, id):
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
                        #print(yolo_output[0][0][0])
                        #print(len(yolo_output[0][0][0]))
                        #yolo_output_new=  np.concatenate(
				        #          ( np.reshape(yolo_output[0][0][0], [-1, 4096]),
						#		    np.reshape([0,0,0,0,0,0], [-1, 6]) ),
						#		  axis = 1)
                        #yolo_output_new= np.reshape(yolo_output_new, 4102)
                        yolo_output_batch.append(yolo_output)
                #yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
                #print 'yolo_output_batch:' + str(yolo_output_batch)
                return yolo_output_batch


        def load_rolo_gt_test(self, path, batch_size, num_steps, id):
                lines= self.load_dataset_gt(path)
                offset= num_steps - 2  # offset is for prediction of the future
                st= id
                ed= id + batch_size*num_steps
                batch_locations= []
                for id in range(st+offset, ed+offset, num_steps):
                        location= self.find_gt_location(lines, id)
                        batch_locations.append(location)
                return batch_locations

#-----------------------------------------------------------------------------------------------
def coordinates_to_heatmap_vec(self, coord):
        heatmap_vec = np.zeros(1024)
        [x1, y1, x2, y2] = coord
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                index = y*32 + x
                heatmap_vec[index] = 1.0   #random.uniform(0.8, 1)#1.0
        return heatmap_vec


def heatmap_vec_to_heatmap(self, heatmap_vec):
    size = 32
    heatmap= np.zeros((size, size))
    for y in range(0, size):
        for x in range(0, size):
            index = y*size + x
            heatmap[y][x] = heatmap_vec[index]
    return heatmap


def draw_heatmap(self, heatmap):
    fig = plt.figure(1, figsize=(10,10))
    ax2 = fig.add_subplot(222)
    #print(heatmap)
    ax2.imshow(heatmap, origin='lower', aspect='auto')
    ax2.set_title("heatmap")
    plt.show()
    #cv2.imshow('YOLO_small detection',heatmap)
    #cv2.waitKey(1)

def load_yolo_output_heatmap(self, fold, batch_size, num_steps, id):
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
                #print(yolo_output)
                yolo_output= np.reshape(yolo_output, 5120)
                yolo_output_batch.append(yolo_output)
        yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 5120])
        return yolo_output_batch

def createFolder( path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_folder( path):
        paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
        return sorted(paths)


def load_dataset_gt(gt_file):
        txtfile = open(gt_file, "r")
        lines = txtfile.read().split('\n')  #'\r\n'
        return lines


def find_gt_location( lines, id):
        line = lines[id]
        elems = line.split('\t')   # for gt type 2
        if len(elems) < 4:
            elems = line.split(',') # for gt type 1
        x1 = elems[0]
        y1 = elems[1]
        w = elems[2]
        h = elems[3]
        gt_location = [int(x1), int(y1), int(w), int(h)]
        return gt_location


def find_yolo_location( fold, id):
        paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        path= paths[id-1]
        #print(path)
        yolo_output = np.load(path)
        #print(yolo_output[0][4096:4102])
        yolo_location= yolo_output[0][4097:4101]
        return yolo_location

import re
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def find_yolo_kalman_location( fold, id):
        paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
        sort_nicely(paths)
        path= paths[id-1]
        f = open(path, 'r')
        yolo_kalman_str = f.read().split(' ')
        yolo_kalman = [float(yolo_kalman_str[0]),float(yolo_kalman_str[1]),float(yolo_kalman_str[2]),float(yolo_kalman_str[3]) ]
        yolo_kalman_location= yolo_kalman[0:4]
        return yolo_kalman_location


def find_rolo_location( fold, id):
        filename= str(id) + '.npy'
        path= os.path.join(fold, filename)
        rolo_output = np.load(path)
        return rolo_output


def file_to_img( filepath):
    img = cv2.imread(filepath)
    return img


def debug_location( img, location):
    img_cp = img.copy()
    x = int(location[1])
    y = int(location[2])
    w = int(location[3])//2
    h = int(location[4])//2
    cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
    cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    cv2.imshow('YOLO_small detection',img_cp)
    cv2.waitKey(1)


def debug_gt_location( img, location):
    img_cp = img.copy()
    x = int(location[0])
    y = int(location[1])
    w = int(location[2])
    h = int(location[3])
    cv2.rectangle(img_cp,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('gt',img_cp)
    cv2.waitKey(1)


def debug_3_locations( img, gt_location, yolo_location, rolo_location):
    img_cp = img.copy()
    for i in range(3):  # b-g-r channels
        if i== 0: location= gt_location; color= (0, 0, 255)       # red for gt
        elif i ==1: location= yolo_location; color= (255, 0, 0)   # blur for yolo
        elif i ==2: location= rolo_location; color= (0, 255, 0)   # green for rolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 1 or i== 2: cv2.rectangle(img_cp,(x-w//2, y-h//2),(x+w//2,y+h//2), color, 2)
        elif i== 0: cv2.rectangle(img_cp,(x,y),(x+w,y+h), color, 2)
    cv2.imshow('3 locations',img_cp)
    cv2.waitKey(100)
    return img_cp


def debug_kalman_locations(img, gt_location, yolo_location):
    img_cp = img.copy()
    for i in range(2):  # b-g-r channels
        if i== 0: location= gt_location; color= (0, 0, 255)       # red for gt
        elif i ==1: location= yolo_location; color= (255, 0, 0)   # blu3 for yolo_kalman
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        cv2.rectangle(img_cp,(x,y),(x+w,y+h), color, 2)
    cv2.imshow('2 locations',img_cp)
    cv2.waitKey(100)
    return img_cp


def debug_2_locations( img, gt_location, yolo_location):
    img_cp = img.copy()
    for i in range(3):  # b-g-r channels
        if i== 0: location= gt_location; color= (0, 0, 255)       # red for gt
        elif i ==1: location= yolo_location; color= (255, 0, 0)   # blur for yolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 1: cv2.rectangle(img_cp,(x-w//2, y-h//2),(x+w//2,y+h//2), color, 2)
        elif i== 0: cv2.rectangle(img_cp,(x,y),(x+w,y+h), color, 2)
    cv2.imshow('2 locations',img_cp)
    cv2.waitKey(100)
    return img_cp

def save_rolo_output(out_fold, rolo_output, filename):
    name_no_ext= os.path.splitext(filename)[0]
    output_name= name_no_ext
    path = os.path.join(out_fold, output_name)
    np.save(path, rolo_output)


def save_rolo_output(out_fold, rolo_output, step, num_steps, batch_size):
        assert(len(rolo_output)== batch_size)
        st= step * batch_size * num_steps - 2
        for i in range(batch_size):
            id = st + (i + 1)* num_steps + 1
            pred = rolo_output[i]
            path = os.path.join(out_fold, str(id))
            np.save(path, pred)


def save_rolo_output_test( out_fold, rolo_output, step, num_steps, batch_size):
        assert(len(rolo_output)== batch_size)
        st= step - 2 #* batch_size * num_steps
        for i in range(batch_size):
            id = st + (i + 1)* num_steps + 1
            pred = rolo_output[i]
            path = os.path.join(out_fold, str(id))
            np.save(path, pred)


def save_rolo_output_heatmap( out_fold, rolo_heat, step, num_steps, batch_size):
        assert(len(rolo_heat)== batch_size)
        st= step - 2 #* batch_size * num_steps
        for i in range(batch_size):
            id = st + (i + 1)* num_steps + 1
            pred = rolo_heat[i]
            path = os.path.join(out_fold, str(id))
            np.save(path, pred)


def locations_normal(wid, ht, locations):
    #print("location in func: ", locations)
    wid *= 1.0
    ht *= 1.0
    locations[0] *= wid
    locations[1] *= ht
    locations[2] *= wid
    locations[3] *= ht
    return locations


def locations_from_0_to_1(wid, ht, locations):
    #print("location in func: ", locations[0][0])
    wid *= 1.0
    ht *= 1.0
    for i in range(len(locations)):
        # convert top-left point (x,y) to mid point (x, y)
        locations[i][0] += locations[i][2] / 2.0
        locations[i][1] += locations[i][3] / 2.0
        # convert to [0, 1]
        locations[i][0] /= wid
        locations[i][1] /= ht
        locations[i][2] /= wid
        locations[i][3] /= ht
    return locations


def validate_box(box):
    for i in range(len(box)):
        if math.isnan(box[i]): box[i] = 0


def iou(box1, box2):
    # Prevent NaN in benchmark results
    validate_box(box1)
    validate_box(box2)

    # change float to int, in order to prevent overflow
    box1 = map(int, box1)
    box2 = map(int, box2)

    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb <= 0 or lr <= 0 :
        intersection = 0
        #print "intersection= 0"
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def iou_0_1(box1, box2, w, h):
    box1 = locations_normal(w,h,box1)
    box2 = locations_normal(w,h,box2)
    #print box1
    #print box2
    return iou(box1,box2)

def cal_rolo_IOU(location, gt_location):
    location[0] = location[0] - location[2]/2
    location[1] = location[1] - location[3]/2
    loss = iou(location, gt_location)
    return loss


def cal_yolo_IOU(location, gt_location):
    # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
    location[0] = location[0] - location[2]/2
    location[1] = location[1] - location[3]/2
    loss = iou(location, gt_location)
    return loss


def cal_benchmark_IOU(location, gt_location):
    loss = iou(location, gt_location)
    return loss

def cal_rolo_score(location, gt_location, thresh):
    rolo_iou = cal_rolo_IOU(location, gt_location)
    if rolo_iou >= thresh:
        score = 1
    else:
        score = 0
    return score


def cal_yolo_score(location, gt_location, thresh):
    yolo_iou = cal_yolo_IOU(location, gt_location)
    if yolo_iou >= thresh:
        score = 1
    else:
        score = 0
    return score


def cal_yolo_kalman_score(location, gt_location, thresh):
    yolo_iou = iou(location, gt_location)
    if yolo_iou >= thresh:
        score = 1
    else:
        score = 0
    return score


def cal_benchmark_score(location, gt_location, thresh):
    benchmark_iou = cal_benchmark_IOU(location, gt_location)
    if benchmark_iou >= thresh:
        score = 1
    else:
        score = 0
    return score


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
                #print(yolo_output)
                yolo_output= np.reshape(yolo_output, 4102)
                yolo_output_batch.append(yolo_output)
        yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
        return yolo_output_batch



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


