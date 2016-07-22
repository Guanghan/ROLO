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
Script File: ROLO_evaluation.py

Description:

	ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking
	Paper: http://arxiv.org/abs/1607.05781
	Author: Guanghan Ning
	Webpage: http://guanghan.info/
'''

import numpy
print numpy.__path__

import cv2
import os
import numpy as np
import sys
import ROLO_utils as utils
import matplotlib.pyplot as plot
import pickle
import scipy.io
import re
import h5py
import matlab.engine

''' -----------------------------Deal with benchmark results: matlab format-------------------------- '''
def choose_benchmark_method(id):
    if id == 0:
        method = 'STRUCK'
    elif id == 1:
        method = 'CXT'
    elif id == 2:
        method = 'TLD'
    elif id == 3:
        method = 'OAB'
    elif id == 4:
        method = 'CSK'
    elif id == 5:
        method = 'RS'
    elif id == 6:
        method = 'LSK'
    elif id == 7:
        method = 'VTD'
    elif id == 8:
        method = 'VTS'
    elif id == 9:
        method = 'CNN-SVM'
    elif id == 10:
        method = 'Staple'
    return method


def choose_mat_file(method_id, sequence_id):
    [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)
    method_name = choose_benchmark_method(method_id)
    mat_file = sequence_name + '_' + method_name + '.mat'

    return mat_file


def load_mat_results(mat_file, TRE, SRE, OPE, id):
    if TRE is True:
        fold = '/u03/Guanghan/dev/ROLO-dev/experiments/benchmark_results/pami15_TRE'
    elif SRE is True:
        fold = '/u03/Guanghan/dev/ROLO-dev/experiments/benchmark_results/pami15_SRE'
    elif OPE is True:
        fold = '/u03/Guanghan/dev/ROLO-dev/experiments/benchmark_results/pami15_TRE'
        id = 0
    mat_path = os.path.join(fold, mat_file)
    CNN_SVM = False
    if CNN_SVM is True:
        eng = matlab.engine.start_matlab()
        content = eng.load(mat_path,nargout=1)
        mat_results= content['results'][0]['res']#[0]
        numbers= [0, content['results'][0]['len']]
        eng.exit()
    else:
        mat = scipy.io.loadmat(mat_path)
        mat_results = mat['results'][0][id][0][0][5]
        mat_range_str = mat['results'][0][id][0][0][2]
        numbers= re.findall(r'\d+', str(mat_range_str))
    return [mat_results, int(numbers[0]), int(numbers[1])]


def load_benchmark_results():
    # 1. read mat file, output numpy file to: e.g., /u03/Guanghan/dev/ROLO-dev/benchmark/DATA/Car1/STRUCK/
    # 2. convert to same format as yolo and rolo
    # 3. evaluate AUC and avg_IOU score, for drawing the success plot
    # 4. Compare with ROLO and YOLO's OPE (3 parts: TRE ,SRE, SRER)

    return


def evaluate_benchmark_avg_IOU(method_id):        # calculate AUC(Average Under Curve) of benchmark algorithms
    ''' PARAMETERS '''
    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1.0

    avg_score= 0

    method_name= choose_benchmark_method(method_id)
    file_name= 'output/IOU/avgIOU_' + method_name + '.txt'
    f=  open(file_name, 'w')

    for sequence_id in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

        # Load benchmark detection loc
        mat_file = choose_mat_file(method_id, sequence_id)
        [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, False, False, True, 0)

        # Load ground truth detection loc
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        lines = utils.load_dataset_gt(gt_file_path)

        #
        total= 0
        total_score= 0

        for id in range(0, ed_frame_num):
            location=  locations[id]
            gt_location = utils.find_gt_location(lines, id)

            score =  utils.iou(location, gt_location)
            total_score += score

            total += 1.0

        total_score /= total

        [dummy, dummy, sequence_name, dummy, dummy]= utils.choose_video_sequence(sequence_id)
        print(method_name, ',' ,sequence_name, ": avg_IOU = ",  total_score)

        f.write(method_name + ', ' + sequence_name + ": avg_IOU = " + str("{:.3f}".format(total_score)) + '\n')

        avg_score += total_score

    f.close()

    avg_score /= num_evaluate
    print('average score over all sequences:', avg_score)


def evaluate_benchmark_AUC_OPE(method_id):        # calculate AUC(Average Under Curve) of benchmark algorithms
    ''' PARAMETERS '''
    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1.0

    AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        print("thresh= ", thresh)
        avg_score= 0

        for sequence_id in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

            # Load benchmark detection loc
            mat_file = choose_mat_file(method_id, sequence_id)

            [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, False, False, True, 0)
            #print(locations)

            # Load ground truth detection loc
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            lines = utils.load_dataset_gt(gt_file_path)

            #
            total= 0
            total_score= 0

            for id in range(0, ed_frame_num):
                location=  locations[id]
                gt_location = utils.find_gt_location(lines, id)

                score =  utils.cal_benchmark_score(location, gt_location, thresh)
                total_score += score

                total += 1.0

            total_score /= total
            avg_score += total_score

        AUC_score.append(avg_score/num_evaluate)
        print("(thresh, AUC_score) = ", thresh, ' ', avg_score/num_evaluate)

    method_name= choose_benchmark_method(method_id)
    file_name= 'output/AUC_score_' + method_name + '.pickle'
    with open(file_name, 'w') as f:
        pickle.dump(AUC_score, f)


def evaluate_benchmark_AUC_TRE(method_id):        # calculate TRE of AUC(Average Under Curve) of benchmark algorithms
    ''' PARAMETERS '''
    evaluate_st = 0
    evaluate_ed = 29
    TRE_num = 20
    num_evaluate= evaluate_ed - evaluate_st + 1.0

    AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        print("thresh= ", thresh)
        avg_score= 0

        for sequence_id in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

            # Load ground truth detection loc
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            lines = utils.load_dataset_gt(gt_file_path)

            # Load benchmark detection loc
            mat_file = choose_mat_file(method_id, sequence_id)

            total_score_over_TREs= 0

            for locations_id in range(0, TRE_num):
                [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, True, False, False, locations_id)

                ct_frames= 0
                total_score_over_frames= 0
                for id in range(st_frame_num-1, ed_frame_num):
                    id_offset= id - st_frame_num + 1
                    location=  locations[id_offset]  # id_offset, not id
                    gt_location = utils.find_gt_location(lines, id)  #id, not id_offset

                    score =  utils.cal_benchmark_score(location, gt_location, thresh)
                    total_score_over_frames += score
                    ct_frames += 1.0

                total_score_over_frames /= ct_frames
                total_score_over_TREs += total_score_over_frames
            total_score_over_TREs /= (TRE_num * 1.0)
            avg_score += total_score_over_TREs

        AUC_score.append(avg_score/num_evaluate)
        print("(thresh, AUC_score) = ", thresh, ' ', avg_score/num_evaluate)

    method_name= choose_benchmark_method(method_id)
    file_name= 'output/TRE_score_' + method_name + '.pickle'
    with open(file_name, 'w') as f:
        pickle.dump(AUC_score, f)


def evaluate_benchmark_avg_IOU_TRE(method_id):        # calculate TRE of AUC(Average Under Curve) of benchmark algorithms
    ''' PARAMETERS '''
    evaluate_st = 0
    evaluate_ed = 29
    TRE_num = 20
    num_evaluate= evaluate_ed - evaluate_st + 1.0

    score_over_sequences= 0

    method_name= choose_benchmark_method(method_id)
    file_name= 'output/IOU/TRE_avgIOU_' + method_name + '.txt'
    f=  open(file_name, 'w')

    for sequence_id in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

        # Load ground truth detection loc
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        lines = utils.load_dataset_gt(gt_file_path)

        # Load benchmark detection loc
        mat_file = choose_mat_file(method_id, sequence_id)

        score_over_TREs= 0
        for locations_id in range(0, TRE_num):
            [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, True, False, False, locations_id)

            ct_frames= 0
            score_over_frames= 0
            for id in range(st_frame_num-1, ed_frame_num):
                id_offset= id - st_frame_num + 1
                location=  locations[id_offset]  # id_offset, not id
                gt_location = utils.find_gt_location(lines, id)  #id, not id_offset

                score =  utils.iou(location, gt_location)
                score_over_frames += score
                ct_frames += 1.0
            score_over_frames /= ct_frames
            score_over_TREs += score_over_frames
        score_over_TREs /= (TRE_num * 1.0)
        score_over_sequences += score_over_TREs

    avg_IOU_TRE_score= score_over_sequences/num_evaluate
    print("avg_IOU_TRE_score = ", avg_IOU_TRE_score)

    f.write(method_name + ', ' + sequence_name + ": TRE_avg_IOU = " + str("{:.3f}".format(avg_IOU_TRE_score)) + '\n')
    f.close()

    return avg_IOU_TRE_score


def evaluate_benchmark_AUC_SRE(method_id):        # calculate TRE of AUC(Average Under Curve) of benchmark algorithms
    ''' PARAMETERS '''
    evaluate_st = 0
    evaluate_ed = 29
    SRE_num = 12
    num_evaluate= evaluate_ed - evaluate_st + 1.0

    AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0 + + 0.0001
        print("thresh= ", thresh)
        avg_score_over_sequences = 0

        for sequence_id in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

            # Load ground truth detection loc
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            lines = utils.load_dataset_gt(gt_file_path)

            # Load benchmark detection loc
            mat_file = choose_mat_file(method_id, sequence_id)

            total= 0
            avg_score= 0

            for locations_id in range(0, SRE_num):
                [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, False, True, False, locations_id)
                total += 1.0
                ct = 0
                total_score= 0
                for id in range(st_frame_num-1, ed_frame_num):
                    id_offset= id - st_frame_num + 1
                    location=  locations[id_offset]  # id_offset, not id
                    gt_location = utils.find_gt_location(lines, id)  #id, not id_offset

                    score =  utils.cal_benchmark_score(location, gt_location, thresh)
                    total_score += score

                    ct += 1.0

                total_score /= ct
                avg_score += total_score
            avg_score /= total
            avg_score_over_sequences += avg_score

        AUC_score.append(avg_score_over_sequences/num_evaluate)
        print("(thresh, AUC_score) = ", thresh, ' ', avg_score_over_sequences/num_evaluate)

    method_name= choose_benchmark_method(method_id)
    file_name= 'output/SRE_score_' + method_name + '.pickle'
    with open(file_name, 'w') as f:
        pickle.dump(AUC_score, f)



def evaluate_benchmark_avg_IOU_SRE(method_id):        # calculate TRE of AUC(Average Under Curve) of benchmark algorithms
    ''' PARAMETERS '''
    evaluate_st = 0
    evaluate_ed = 29
    SRE_num = 12
    num_evaluate= evaluate_ed - evaluate_st + 1.0

    method_name= choose_benchmark_method(method_id)
    file_name= 'output/IOU/SRE_avgIOU_' + method_name + '.txt'
    f=  open(file_name, 'w')
    avg_score= 0

    for sequence_id in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

        # Load ground truth detection loc
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        lines = utils.load_dataset_gt(gt_file_path)

        # Load benchmark detection loc
        mat_file = choose_mat_file(method_id, sequence_id)

        total= 0
        total_score= 0

        for locations_id in range(0, SRE_num):
            [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, False, True, False, locations_id)

            for id in range(st_frame_num-1, ed_frame_num):
                id_offset= id - st_frame_num + 1
                location=  locations[id_offset]  # id_offset, not id
                gt_location = utils.find_gt_location(lines, id)  #id, not id_offset

                score =  utils.iou(location, gt_location)
                total_score += score

                total += 1.0

        total_score /= total
        avg_score += total_score

    avg_IOU_SRE_score= avg_score/num_evaluate
    print("avg_IOU_score_SRE: ", avg_IOU_SRE_score)

    f.write(method_name + ', ' + sequence_name + ": SRE_avg_IOU = " + str("{:.3f}".format(avg_IOU_SRE_score)) + '\n')
    f.close()

    return avg_IOU_SRE_score




''' -----------------------------Deal with ROLO results: python format-----------------------------'''
def draw_AUC_OPE():

    num_methods = 9 + 1

    with open('output/AUC_score.pickle') as f:
        [yolo_AUC_score, rolo_AUC_score] = pickle.load(f)
    yolo_AUC_score.append(0)
    rolo_AUC_score.append(0)
    yolo_AUC_score = np.asarray(yolo_AUC_score)
    rolo_AUC_score = np.asarray(rolo_AUC_score)

    with open('output/AUC_kalman_score.pickle') as f:
        [yolo_kalman_AUC_score] = pickle.load(f)
    yolo_kalman_AUC_score.append(0)
    yolo_kalman_AUC_score = np.asarray(yolo_kalman_AUC_score)


    benchmark_AUC_score = []
    for method_id in range(0, num_methods):
        method_name= choose_benchmark_method(method_id)
        file_name= 'output/AUC_score_' + method_name + '.pickle'
        with open(file_name) as f:
            AUC_score = pickle.load(f)
            AUC_score.append(0)
            AUC_score = np.asarray(AUC_score)
            benchmark_AUC_score.append(AUC_score)


    x = [i/100.0 for i in range(0, 105, 5)]

    print(len(x))
    print(len(yolo_AUC_score))

    print(x)
    print(yolo_AUC_score)
    print(rolo_AUC_score)

    fig= plot.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 100, 10))
    plot.title("Success Plot of OPE")
    #plot.title("Success Plot of OPE30: AUC(Average Under Curve)")
    plot.xlabel("overlap threshold")
    plot.ylabel("success rate")
    '''
    plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO", linestyle='-', marker= "s", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD", linestyle='--', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS", linestyle='-', marker= "o", markersize= 5, linewidth= 1, markevery= 1)
    '''

    'test all 30'
    # #plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.564]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)   #exp all frames
    # plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.458]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)  #exp 1/3 frames
    # #plot.plot(x, benchmark_AUC_score[9]*100, color = 'y', label = "CNN-SVM[0.520]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    # #plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO [0.440]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK [0.410]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB [0.366]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK [0.356]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD [0.343]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    #
    # plot.plot(x, yolo_kalman_AUC_score*100, color = 'k', label = "YOLO+SORT [0.341]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    #
    # plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT [0.333]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS [0.325]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS [0.320]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD [0.315]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    # plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK [0.311]", linestyle='-', markersize= 5, linewidth= 2, markevery= 1)





    '''test last 8'''
    plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.476]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    #plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO [0.459]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK [0.454]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS [0.444]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD [0.433]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT [0.433]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK [0.428]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, yolo_kalman_AUC_score*100, color = 'k', label = "YOLO+SORT [0.406]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK [0.406]", linestyle='-', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS [0.392]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB [0.366]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD [0.318]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)



    #plot.plot(x, benchmark_AUC_score[9]*100, color = 'y', label = "VTS", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.axis([0, 1, 0, 100])

    plot.grid()
    plot.legend(loc = 1, prop={'size':10})
    plot.show()


def draw_AUC_TRE():

    with open('output/AUC_score_TRE.pickle') as f:
        [yolo_AUC_score, rolo_AUC_score] = pickle.load(f)
    yolo_AUC_score.append(0)
    rolo_AUC_score.append(0)
    yolo_AUC_score = np.asarray(yolo_AUC_score)
    rolo_AUC_score = np.asarray(rolo_AUC_score)

    with open('output/AUC_kalman_score_TRE.pickle') as f:
        [yolo_kalman_AUC_score] = pickle.load(f)
    yolo_kalman_AUC_score.append(0)
    yolo_kalman_AUC_score = np.asarray(yolo_kalman_AUC_score)

    benchmark_AUC_score = []
    for method_id in range(0, 9):
        method_name= choose_benchmark_method(method_id)
        file_name= 'output/TRE_score_' + method_name + '.pickle'
        with open(file_name) as f:
            AUC_score = pickle.load(f)
            AUC_score.append(0)
            AUC_score = np.asarray(AUC_score)
            benchmark_AUC_score.append(AUC_score)

    x = [i/100.0 for i in range(0, 105, 5)]

    print(len(x))
    print(len(yolo_AUC_score))

    print(x)
    print(yolo_AUC_score)
    print(rolo_AUC_score)

    fig= plot.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 100, 10))
    plot.title("Success Plot of TRE")
    plot.xlabel("overlap threshold")
    plot.ylabel("success rate")

    '''test all 30'''
    plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.562]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK [0.548]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB [0.462]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK [0.459]", linestyle='-', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT [0.432]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    #plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO [0.429]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK [0.427]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS [0.425]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD [0.414]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD [0.414]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS [0.397]", linestyle= '-',  markersize= 5, linewidth= 2, markevery= 1)

    plot.plot(x, yolo_kalman_AUC_score*100, color = 'k', label = "YOLO+SORT [0.322]", linestyle= '--',  markersize= 5, linewidth= 2, markevery= 1)

    plot.axis([0, 1, 0, 100])

    plot.grid()
    plot.legend(loc = 1, prop={'size':10})
    plot.show()


def draw_AUC_SRE():

    with open('output/AUC_score.pickle') as f:
        [yolo_AUC_score, rolo_AUC_score] = pickle.load(f)
    yolo_AUC_score.append(0)
    rolo_AUC_score.append(0)
    yolo_AUC_score = np.asarray(yolo_AUC_score)
    rolo_AUC_score = np.asarray(rolo_AUC_score)

    with open('output/AUC_kalman_score.pickle') as f:
        [yolo_kalman_AUC_score] = pickle.load(f)
    yolo_kalman_AUC_score.append(0)
    yolo_kalman_AUC_score = np.asarray(yolo_kalman_AUC_score)


    benchmark_AUC_score = []
    for method_id in range(0, 9):
        method_name= choose_benchmark_method(method_id)
        file_name= 'output/SRE_score_' + method_name + '.pickle'
        with open(file_name) as f:
            AUC_score = pickle.load(f)
            AUC_score.append(0)
            AUC_score = np.asarray(AUC_score)
            benchmark_AUC_score.append(AUC_score)

    x = [i/100.0 for i in range(0, 105, 5)]

    print(len(x))
    print(len(yolo_AUC_score))

    print(x)
    print(yolo_AUC_score)
    print(rolo_AUC_score)

    fig= plot.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 100, 10))
    plot.title("Success Plot of SRE")
    plot.xlabel("overlap threshold")
    plot.ylabel("success rate")

    plot.plot(x, rolo_AUC_score*100, color = 'g', label = "ROLO [0.564]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    #plot.plot(x, yolo_AUC_score*100, color = 'g', label = "YOLO [0.440]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[0]*100, color = 'r', label = "STRUCK [0.391]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, yolo_kalman_AUC_score*100, color = 'k', label = "YOLO+SORT [0.341]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[3]*100, color = 'b', label = "OAB [0.341]", linestyle='--', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[2]*100, color = 'b', label = "TLD [0.331]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[5]*100, color = 'c', label = "RS [0.320]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[6]*100, color = 'm', label = "LSK [0.302]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[1]*100, color = 'r', label = "CXT [0.295]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[4]*100, color = 'c', label = "CSK [0.295]", linestyle='-', markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[7]*100, color = 'm', label = "VTD [0.286]", linestyle='--',  markersize= 5, linewidth= 2, markevery= 1)
    plot.plot(x, benchmark_AUC_score[8]*100, color = 'y', label = "VTS [0.284]", linestyle='-',  markersize= 5, linewidth= 2, markevery= 1)


    plot.axis([0, 1, 0, 100])

    plot.grid()
    plot.legend(loc = 1, prop={'size':10})
    plot.show()


def draw_step_IOU_curve():

    #x = [i for i in range(3, 11, 3)]
    x= np.asarray([1, 3, 6, 9])
    avg_IOU = np.asarray([0.359, 0.434, 0.458, 0.427])

    fig= plot.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_yticks(np.arange(0.35, 0.47, 0.02))
    plot.title("The average accuracy over the numbers of steps")
    plot.xlabel("step")
    plot.ylabel("Accuracy [IoU]")

    plot.plot(x, avg_IOU, color = 'g', linestyle='-',  marker= "s", markersize= 10, linewidth= 2, markevery= 1)
    plot.axis([1, 10, 0.35, 0.47])

    plot.grid()
    plot.legend(loc = 1, prop={'size':10})
    plot.show()


def draw_step_fps_curve():

    avg_fps = np.asarray([271, 110, 61, 42])
    x= np.asarray([1, 3, 6, 9])
    #x = [i for i in range(3, 11, 3)]
    print x

    fig= plot.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_yticks(np.arange(0, 275, 30))
    plot.title("Fps of the tracking module over the numbers of steps")
    plot.xlabel("step")
    plot.ylabel("Frames Per Second (fps)")

    plot.plot(x, avg_fps, color = 'r', linestyle='-',  marker= "^", markersize= 10, linewidth= 2, markevery= 1)
    plot.axis([1, 10, 20, 275])

    plot.grid()
    plot.legend(loc = 1, prop={'size':10})
    plot.show()


def evaluate_AUC_TRE():        # calculate AUC(Average Under Curve) TRE
    ''' PARAMETERS '''
    num_steps= 3
    TRE_num = 20

    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    yolo_AUC_score= []
    rolo_AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        #print("thresh= ", thresh)
        rolo_avg_score= 0
        yolo_avg_score= 0

        for sequence_id in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

            img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
            rolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')

            paths_imgs = utils.load_folder( img_fold_path)
            paths_rolo= utils.load_folder( rolo_out_path)
            lines = utils.load_dataset_gt( gt_file_path)

            # Define the codec and create VideoWriter object
            rolo_total_score_over_TREs= 0
            yolo_total_score_over_TREs= 0

            # Load benchmark detection loc
            mat_file = choose_mat_file(0, sequence_id)

            for locations_id in range(0, TRE_num):
                [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, True, False, False, locations_id)
                print(st_frame_num)
                ct_frames= 0
                rolo_total_score_over_frames= 0
                yolo_total_score_over_frames= 0
                for i in range(st_frame_num-1, len(paths_rolo)- num_steps):
                    id= i + 1
                    test_id= id + num_steps

                    yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
                    yolo_location= utils.locations_normal(wid, ht, yolo_location)

                    rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
                    rolo_location = utils.locations_normal( wid, ht, rolo_location)

                    gt_location = utils.find_gt_location( lines, test_id - 1)

                    rolo_score = utils.cal_rolo_score(rolo_location, gt_location, thresh)
                    rolo_total_score_over_frames += rolo_score
                    yolo_score =  utils.cal_yolo_score(yolo_location, gt_location, thresh)
                    yolo_total_score_over_frames  += yolo_score
                    ct_frames += 1.0
                rolo_total_score_over_frames /= ct_frames
                yolo_total_score_over_frames /= ct_frames
                rolo_total_score_over_TREs += rolo_total_score_over_frames
                yolo_total_score_over_TREs += yolo_total_score_over_frames

            rolo_total_score_over_TREs /= (TRE_num * 1.0)
            yolo_total_score_over_TREs /= (TRE_num * 1.0)

            rolo_avg_score += rolo_total_score_over_TREs
            yolo_avg_score += yolo_total_score_over_TREs

            print('Sequence ID: ', sequence_id)
            print("yolo_avg_score = ", yolo_total_score_over_TREs)
            print("rolo_avg_score = ", rolo_total_score_over_TREs)

        yolo_AUC_score.append(yolo_avg_score/num_evaluate)
        rolo_AUC_score.append(rolo_avg_score/num_evaluate)

        print("(thresh, yolo_AUC_score) = ", thresh, ' ', yolo_avg_score/num_evaluate)
        print("(thresh, rolo_AUC_score) = ", thresh, ' ', rolo_avg_score/num_evaluate)

    with open('output/AUC_score_TRE.pickle', 'w') as f:
        pickle.dump([yolo_AUC_score, rolo_AUC_score], f)


def evaluate_kalman_AUC_TRE():        # calculate AUC(Average Under Curve) TRE
    ''' PARAMETERS '''
    num_steps= 3
    TRE_num = 20

    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    yolo_AUC_score= []
    rolo_AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        #print("thresh= ", thresh)
        yolo_avg_score= 0

        for sequence_id in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

            img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_output_kalman_txt/')

            paths_imgs = utils.load_folder( img_fold_path)
            paths_yolo= utils.load_folder( yolo_out_path)

            lines = utils.load_dataset_gt( gt_file_path)

            # Define the codec and create VideoWriter object
            yolo_total_score_over_TREs= 0

            # Load benchmark detection loc
            mat_file = choose_mat_file(0, sequence_id)

            for locations_id in range(0, TRE_num):
                [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, True, False, False, locations_id)
                #print(st_frame_num)
                ct_frames= 0
                yolo_total_score_over_frames= 0

                for i in range(st_frame_num-1, len(paths_yolo)- num_steps):
                    id= i + 1
                    test_id= id + num_steps

                    yolo_location= utils.find_yolo_kalman_location(yolo_out_path, test_id)
                    gt_location = utils.find_gt_location( lines, test_id - 1)

                    yolo_score =  utils.cal_yolo_kalman_score(yolo_location, gt_location, thresh)
                    yolo_total_score_over_frames  += yolo_score
                    ct_frames += 1.0
                if ct_frames!= 0: yolo_total_score_over_frames /= ct_frames
                yolo_total_score_over_TREs += yolo_total_score_over_frames

            yolo_total_score_over_TREs /= (TRE_num * 1.0)
            yolo_avg_score += yolo_total_score_over_TREs

            print('Sequence ID: ', sequence_id)
            print("yolo_avg_score = ", yolo_total_score_over_TREs)

        yolo_AUC_score.append(yolo_avg_score/num_evaluate)
        print("(thresh, yolo_AUC_score) = ", thresh, ' ', yolo_avg_score/num_evaluate)

    with open('output/AUC_kalman_score_TRE.pickle', 'w') as f:
        pickle.dump([yolo_AUC_score], f)


def evaluate_AUC():        # calculate AUC(Average Under Curve)
    ''' PARAMETERS '''
    num_steps= 3

    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    yolo_AUC_score= []
    rolo_AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        print("thresh= ", thresh)
        rolo_avg_score= 0
        yolo_avg_score= 0

        for test in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(test)

            img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
            rolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')

            print(rolo_out_path)

            paths_imgs = utils.load_folder( img_fold_path)
            paths_rolo= utils.load_folder( rolo_out_path)
            lines = utils.load_dataset_gt( gt_file_path)

            # Define the codec and create VideoWriter object
            total= 0
            rolo_total_score= 0
            yolo_total_score= 0

            for i in range(len(paths_rolo)- num_steps):
                id= i + 1
                test_id= id + num_steps

                #path = paths_imgs[test_id]
                #img = utils.file_to_img(None, path)

                #if(img is None): break

                yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
                yolo_location= utils.locations_normal(wid, ht, yolo_location)

                rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
                rolo_location = utils.locations_normal( wid, ht, rolo_location)

                gt_location = utils.find_gt_location( lines, test_id - 1)

                rolo_score = utils.cal_rolo_score(rolo_location, gt_location, thresh)
                #print('rolo_score', rolo_score)
                rolo_total_score += rolo_score
                #print('rolo_total_score', rolo_total_score)
                yolo_score =  utils.cal_yolo_score(yolo_location, gt_location, thresh)
                yolo_total_score += yolo_score
                total += 1.0

            rolo_total_score /= total
            yolo_total_score /= total

            rolo_avg_score += rolo_total_score
            yolo_avg_score += yolo_total_score

            print('Sequence ID: ', test)
            print("yolo_avg_score = ", yolo_total_score)
            print("rolo_avg_score = ", rolo_total_score)

        yolo_AUC_score.append(yolo_avg_score/num_evaluate)
        rolo_AUC_score.append(rolo_avg_score/num_evaluate)

        print("(thresh, yolo_AUC_score) = ", thresh, ' ', yolo_avg_score/num_evaluate)
        print("(thresh, rolo_AUC_score) = ", thresh, ' ', rolo_avg_score/num_evaluate)

    with open('output/AUC_score.pickle', 'w') as f:
        pickle.dump([yolo_AUC_score, rolo_AUC_score], f)
    #draw_AUC()


def evaluate_kalman_AUC():        # calculate AUC(Average Under Curve)
    ''' PARAMETERS '''
    num_steps= 3

    evaluate_st = 20
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    yolo_AUC_score= []
    for thresh_int in range(0, 100, 5):
        thresh = thresh_int / 100.0  + 0.0001
        print("thresh= ", thresh)
        yolo_avg_score= 0

        for test in range(evaluate_st, evaluate_ed + 1):

            [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(test)

            img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
            gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
            yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_output_kalman_txt/')

            print(yolo_out_path)

            paths_rolo= utils.load_folder( yolo_out_path)
            lines = utils.load_dataset_gt( gt_file_path)

            # Define the codec and create VideoWriter object
            total= 0
            yolo_total_score= 0

            for i in range(len(paths_rolo)- num_steps):
                id= i + 1
                test_id= id + num_steps

                #path = paths_imgs[test_id]
                #img = utils.file_to_img(None, path)

                #if(img is None): break

                yolo_location= utils.find_yolo_kalman_location(yolo_out_path, test_id)
                #yolo_location= utils.locations_normal(wid, ht, yolo_location)

                gt_location = utils.find_gt_location( lines, test_id - 1)

                yolo_score =  utils.cal_yolo_kalman_score(yolo_location, gt_location, thresh)
                yolo_total_score += yolo_score
                total += 1.0

            yolo_total_score /= total
            yolo_avg_score += yolo_total_score

            print('Sequence ID: ', test)
            print("yolo_avg_score = ", yolo_total_score)

        yolo_AUC_score.append(yolo_avg_score/num_evaluate)

        print("(thresh, yolo_kalman_AUC_score) = ", thresh, ' ', yolo_avg_score/num_evaluate)

    with open('output/AUC_kalman_score.pickle', 'w') as f:
        pickle.dump([yolo_AUC_score], f)
    #draw_AUC()


def evaluate_avg_IOU():    # calculate AOS(Average Overlap Score) for each sequence
    ''' PARAMETERS '''
    num_steps= 3
    output_video = False
    display_video = False

    evaluate_st = 0
    evaluate_ed = 29
    yolo_ious = []
    rolo_ious = []

    for test in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(test)

        img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
        rolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')

        print(rolo_out_path)

        paths_imgs = utils.load_folder( img_fold_path)
        paths_rolo= utils.load_folder( rolo_out_path)
        lines = utils.load_dataset_gt( gt_file_path)

        # Define the codec and create VideoWriter object
        fourcc= cv2.cv.CV_FOURCC(*'DIVX')
        video_name = sequence_name + '_test.avi'
        video_path = os.path.join('output/videos/', video_name)
        if output_video is True: video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))

        total= 0
        rolo_avgloss= 0
        yolo_avgloss= 0
        for i in range(len(paths_rolo)- num_steps-1):
            id= i + 1
            test_id= id + num_steps #* num_steps + 1

            path = paths_imgs[test_id]
            img = utils.file_to_img( path)

            if(img is None): break

            yolo_location= utils.find_yolo_location( yolo_out_path, test_id)
            yolo_location= utils.locations_normal(wid, ht, yolo_location)
            #print(yolo_location)

            rolo_location= utils.find_rolo_location(rolo_out_path, test_id)
            rolo_location = utils.locations_normal(wid, ht, rolo_location)
            #print(rolo_location)

            gt_location = utils.find_gt_location(lines, test_id - 1)
            #print('gt: ' + str(test_id))
            #print(gt_location)

            if display_video is True: frame = utils.debug_3_locations(img, gt_location, yolo_location, rolo_location)
            if output_video is True: video.write(frame)
            #cv2.imshow('frame',frame)
            #cv2.waitKey(100)

            rolo_loss = utils.cal_rolo_IOU(rolo_location, gt_location)
            rolo_avgloss += rolo_loss
            yolo_loss=  utils.cal_yolo_IOU(yolo_location, gt_location)
            yolo_avgloss += yolo_loss
            total += 1

        rolo_avgloss /= total
        yolo_avgloss /= total
        print('Sequence ID: ', test)
        print("yolo_avg_iou = ", yolo_avgloss)
        print("rolo_avg_iou = ", rolo_avgloss)

        yolo_ious.append(yolo_avgloss)
        rolo_ious.append(rolo_avgloss)

        if output_video is True: video.release()
        #cv2.destroyAllWindows()

    print('yolo_ious: ', yolo_ious)
    print('rolo_ious: ', rolo_ious)
    log_file = open("output/testing-log-final.txt", "a")
    log_file.write('YOLO_avg_IOU: ')
    for item in range(len(yolo_ious)):
        log_file.write(str("{:.3f}".format(yolo_ious[item])) + '  ')
    log_file.write('\nROLO_avg_IOU: ')
    for item in range(len(rolo_ious)):
        log_file.write(str("{:.3f}".format(rolo_ious[item])) + '  ')
    log_file.write('\n\n')

    yolo_avg_iou = np.mean(yolo_ious)
    rolo_avg_iou = np.mean(rolo_ious)

    log_file.write('YOLO_total_avg_IOU: ')
    log_file.write(str("{:.3f}".format(yolo_avg_iou))+ '  ')
    log_file.write('ROLO_total_avg_IOU: ')
    log_file.write(str("{:.3f}".format(rolo_avg_iou)) + '  ')


def evaluate_avg_IOU_kalman():    # calculate AOS(Average Overlap Score) for each sequence
    ''' PARAMETERS '''
    num_steps= 3
    output_video = False
    display_video = False

    evaluate_st = 20
    evaluate_ed = 29
    yolo_ious = []

    for test in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(test)

        img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        yolo_kalman_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_output_kalman_txt/')

        paths_imgs = utils.load_folder( img_fold_path)
        paths_yolo= utils.load_folder( yolo_kalman_path)
        lines = utils.load_dataset_gt( gt_file_path)

        # Define the codec and create VideoWriter object
        fourcc= cv2.cv.CV_FOURCC(*'DIVX')
        video_name = sequence_name + '_test.avi'
        video_path = os.path.join('output/videos_kalman/', video_name)
        if output_video is True: video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))

        total= 0
        yolo_avgloss= 0
        for i in range(len(paths_yolo)- num_steps-1):
            id= i + 1
            test_id= id + num_steps #* num_steps + 1

            path = paths_imgs[test_id]
            img = utils.file_to_img( path)

            if(img is None): break

            yolo_location= utils.find_yolo_kalman_location( yolo_kalman_path, test_id)
            #yolo_location= utils.locations_normal(wid, ht, yolo_location)
            #print(yolo_location)

            gt_location = utils.find_gt_location(lines, test_id - 1)
            #print('gt: ' + str(test_id))
            #print(gt_location)

            if display_video is True:
                frame = utils.debug_kalman_locations(img, gt_location, yolo_location)
                cv2.imshow('frame',frame)
                cv2.waitKey(100)
            if output_video is True: video.write(frame)

            yolo_loss = utils.iou(yolo_location, gt_location)
            #yolo_loss=  utils.cal_yolo_IOU(yolo_location, gt_location)
            yolo_avgloss += yolo_loss
            total += 1

        yolo_avgloss /= total
        print('Sequence ID: ', test)
        print("yolo_avg_iou = ", yolo_avgloss)

        yolo_ious.append(yolo_avgloss)

        if output_video is True: video.release()
        #cv2.destroyAllWindows()

    print('yolo_ious: ', yolo_ious)
    log_file = open("output/yolo_kalman_log.txt", "a")
    log_file.write('YOLO_avg_IOU: ')
    for item in range(len(yolo_ious)):
        log_file.write(str("{:.3f}".format(yolo_ious[item])) + '  ')
    log_file.write('\n\n')

    yolo_avg_iou = np.mean(yolo_ious)
    log_file.write('YOLO_total_avg_IOU: ')
    log_file.write(str("{:.3f}".format(yolo_avg_iou))+ '  ')


def evaluate_avg_IOU_TRE():        # calculate AUC(Average Under Curve) TRE
    ''' PARAMETERS '''
    num_steps= 3
    TRE_num = 20

    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    rolo_avg_score_over_sequences= 0
    yolo_avg_score_over_sequences= 0

    for sequence_id in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

        img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        yolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
        rolo_out_path= os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')

        paths_imgs = utils.load_folder( img_fold_path)
        paths_rolo= utils.load_folder( rolo_out_path)
        lines = utils.load_dataset_gt( gt_file_path)

        # Define the codec and create VideoWriter object
        rolo_total_score_over_TREs= 0
        yolo_total_score_over_TREs= 0

        # Load benchmark detection loc
        mat_file = choose_mat_file(0, sequence_id)

        for locations_id in range(0, TRE_num):
            #print(locations_id)
            [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, True, False, False, locations_id)
            #print(ed_frame_num)
            ct_frames = 0
            rolo_score_over_interval= 0
            yolo_score_over_interval= 0

            for i in range(st_frame_num-1, len(paths_rolo)- num_steps):
                id= i + 1
                test_id= id + num_steps

                yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
                yolo_location= utils.locations_normal(wid, ht, yolo_location)

                rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
                rolo_location = utils.locations_normal( wid, ht, rolo_location)

                gt_location = utils.find_gt_location( lines, test_id - 1)

                rolo_score = utils.cal_rolo_IOU(rolo_location, gt_location)
                rolo_score_over_interval += rolo_score
                yolo_score =  utils.cal_yolo_IOU(yolo_location, gt_location)
                yolo_score_over_interval += yolo_score
                ct_frames += 1.0

            rolo_score_over_interval /= ct_frames
            yolo_score_over_interval /= ct_frames
            rolo_total_score_over_TREs += rolo_score_over_interval
            yolo_total_score_over_TREs += yolo_score_over_interval
        rolo_total_score_over_TREs /= (TRE_num * 1.0)
        yolo_total_score_over_TREs /= (TRE_num * 1.0)
        print('Sequence ID: ', sequence_id)
        print("yolo_avg_score = ", rolo_total_score_over_TREs)
        print("rolo_avg_score = ", yolo_total_score_over_TREs)
        rolo_avg_score_over_sequences += rolo_total_score_over_TREs
        yolo_avg_score_over_sequences += yolo_total_score_over_TREs

    yolo_avg_IOU_TRE = yolo_avg_score_over_sequences/num_evaluate
    rolo_avg_IOU_TRE = rolo_avg_score_over_sequences/num_evaluate

    print("(yolo_avg_IOU_TRE) = ",  yolo_avg_IOU_TRE)
    print("(rolo_avg_IOU_TRE) = ",  rolo_avg_IOU_TRE)

    log_file = open("output/IOU/avg_IOU_TRE.txt", "a")
    log_file.write('yolo_avg_IOU_TRE: ')
    log_file.write(str("{:.3f}".format(yolo_avg_IOU_TRE)) + '  ')
    log_file.write('\n rolo_avg_IOU_TRE: ')
    log_file.write(str("{:.3f}".format(rolo_avg_IOU_TRE)) + '  ')
    log_file.write('\n\n')


def evaluate_avg_IOU_kalman_TRE():        # calculate AUC(Average Under Curve) TRE
    ''' PARAMETERS '''
    num_steps= 3
    TRE_num = 20

    evaluate_st = 0
    evaluate_ed = 29
    num_evaluate= evaluate_ed - evaluate_st + 1

    yolo_avg_score_over_sequences= 0

    for sequence_id in range(evaluate_st, evaluate_ed + 1):

        [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(sequence_id)

        img_fold_path = os.path.join('benchmark/DATA', sequence_name, 'img/')
        gt_file_path= os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
        yolo_kalman_path= os.path.join('benchmark/DATA', sequence_name, 'yolo_output_kalman_txt/')

        paths_imgs = utils.load_folder( img_fold_path)
        paths_yolo= utils.load_folder( yolo_kalman_path)
        lines = utils.load_dataset_gt( gt_file_path)

        # Define the codec and create VideoWriter object
        yolo_total_score_over_TREs= 0

        # Load benchmark detection loc
        mat_file = choose_mat_file(0, sequence_id)

        for locations_id in range(0, TRE_num):
            #print(locations_id)
            [locations, st_frame_num, ed_frame_num] = load_mat_results(mat_file, True, False, False, locations_id)
            #print(ed_frame_num)
            ct_frames = 0
            yolo_score_over_interval= 0

            for i in range(st_frame_num-1, len(paths_yolo)- num_steps):
                id= i + 1
                test_id= id + num_steps
                #print(test_id)

                yolo_location= utils.find_yolo_kalman_location(yolo_kalman_path, test_id)
                gt_location = utils.find_gt_location( lines, test_id - 1)

                #yolo_score =  utils.cal_yolo_kalman_IOU(yolo_location, gt_location)
                yolo_score = utils.iou(yolo_location, gt_location)
                #print(yolo_score)
                yolo_score_over_interval += yolo_score
                ct_frames += 1.0

            if ct_frames!= 0: yolo_score_over_interval /= ct_frames
            yolo_total_score_over_TREs += yolo_score_over_interval
        yolo_total_score_over_TREs /= (TRE_num * 1.0)
        print('Sequence ID: ', sequence_id)
        print("yolo_avg_score = ", yolo_total_score_over_TREs)
        yolo_avg_score_over_sequences += yolo_total_score_over_TREs

    yolo_avg_IOU_TRE = yolo_avg_score_over_sequences/num_evaluate

    print("(yolo_avg_IOU_TRE) = ",  yolo_avg_IOU_TRE)

    log_file = open("output/IOU/avg_kalman_IOU_TRE.txt", "a")
    log_file.write('yolo_kalman_avg_IOU_TRE: ')
    log_file.write(str("{:.3f}".format(yolo_avg_IOU_TRE)) + '  ')
    log_file.write('\n\n')
'''----------------------------------------main-----------------------------------------------------'''
def main(argv):
    #evaluate_avg_IOU()
    #evaluate_avg_IOU_TRE()
    #evaluate_avg_IOU_kalman()
    #evaluate_avg_IOU_kalman_TRE()

    #evaluate_AUC()   #AUC_OPE and AUC_SRE is the same for ROLO and YOLO
    #evaluate_AUC_TRE()
    #evaluate_kalman_AUC()
    #evaluate_kalman_AUC_TRE()

    #for method_id in range(9, 10):
    #    evaluate_benchmark_avg_IOU(method_id)

    #for method_id in range(0, 9):
    #    evaluate_benchmark_avg_IOU_TRE(method_id)

    #for method_id in range(0, 9):
    #    evaluate_benchmark_avg_IOU_SRE(method_id)

    #for method_id in range(9, 10):
    #   evaluate_benchmark_AUC_OPE(method_id)

    #for method_id in range(0, 9):
    #    evaluate_benchmark_AUC_TRE(method_id)

    #for method_id in range(0, 9):
    #    evaluate_benchmark_AUC_SRE(method_id)

    draw_AUC_OPE()
    #draw_AUC_TRE()
    #draw_AUC_SRE()

    #draw_step_IOU_curve()
    #draw_step_fps_curve()




if __name__=='__main__':
	main(sys.argv)
