
# There are 3 kinds of representation of coordinates
# 1. Coord_decimal: (x0, y0, w, h) all represented in a float between [0, 1], ratio to image width and height, respectively.
#                   (x0, y0) is the middle point of the bounding box.
#                   Used by YOLO output, ROLO input and output.
# 2. Coord_regular: (X1, Y1, W, H) all represented by pixel values in int.
#                   (X1, Y1) is the top-left point of the bounding box
#                   Usually the ground truth box that is read from files are of this format
# 3. Detection in Vector: [4096-d feature_vector] + (class, x0, y0, w, h, prob).
#                   The same as Coord_decimal except that there are more information in the detection
#-----------------------------------------------------------------------------------------------
from utils_io_folder import get_immediate_childfile_paths
import numpy as np
import os

''' 1. I/O with numpy '''

''' 1.1 Save'''
def save_vec_as_numpy_by_frame_id(output_folder_path, frame_id, output_vec):
    filename = str(frame_id)
    save_vec_as_numpy_by_name(output_folder_path, filename, output_vec)

def save_vec_as_numpy_by_name(output_folder_path, filename, output_vec):
    filename_without_ext = os.path.splitext(filename)[0]
    output_file_path = os.path.join(output_folder_path, filename_without_ext)
    np.save(output_file_path, output_vec)


''' 1.2 Load '''
def batchload_yolovecs_from_numpy_folders(batch_folders_paths, batch_frame_ids, batchsize, nsteps):
    batch_vecs = batchload_vecs_from_numpy_folders(batch_folders_paths, batch_frame_ids, batchsize, nsteps)

    if batch_vecs is not False and batch_vecs != -1:
        for vec in batch_vecs:
            vec[0][:][4096] = 0
            vec[0][:][4101] = 0
    return batch_vecs


def batchload_vecs_from_numpy_folders(batch_folders_paths, batch_frame_ids, batchsize, nsteps):
    try:
        assert(len(batch_folders_paths) == batchsize)
    except AssertionError:
        print("\t Not enough pairs to form a minibatch, skip")
        return -1

    batch_vecs = []
    for ct, folder_path in enumerate(batch_folders_paths):
        frame_id = int(batch_frame_ids[ct])
        nsteps_vecs = load_vecs_of_stepsize_in_numpy_folder(folder_path, frame_id, nsteps)
        batch_vecs.append(nsteps_vecs)

    try:
        test_vecs = np.reshape(batch_vecs, [batchsize * nsteps, 4102])
        return batch_vecs
    except ValueError:
        print("\t Not enough frames in video (it's ok), skipped this minibatch")
        return False


def load_vecs_of_stepsize_in_numpy_folder(folder_path, frame_id, nsteps):
    file_paths = get_file_paths_of_stepsize_in_numpy_folder(folder_path, frame_id, nsteps)
    nsteps_vecs = []
    for file_path in file_paths:
        vec_from_file = load_vec_from_numpy_file(file_path)
        nsteps_vecs.append(vec_from_file)
    return nsteps_vecs


def get_file_paths_of_stepsize_in_numpy_folder(folder_path, frame_id, nsteps):
    file_paths = get_immediate_childfile_paths(folder_path)
    [st, ed] = get_range_of_stepsize_by_frame_id(nsteps, frame_id)
    file_paths_batch = file_paths[st:ed]
    return file_paths_batch


def load_vec_from_numpy_file(file_path):
    vec_from_file = np.load(file_path)
    vec_from_file = np.reshape(vec_from_file, 4102)
    return vec_from_file


def batchload_decimal_coords_from_vecs(batch_vecs):
    batch_coords = [vec[4097:4101] for vec in batch_vecs]
    return batch_coords


def load_decimal_coord_from_vec(vec_from_file):
    coord_decimal = vec_from_file[4097:4101]
    return coord_decimal


def get_range_of_stepsize_by_frame_id(nsteps, frame_id, offset= 0):
    [st, ed] = [frame_id, frame_id + nsteps]
    st_ed_range = [st + offset, ed + offset]
    return st_ed_range


''' 2. I/O with text file '''

def load_lines_from_txt_file(txt_file_path):
    with open(txt_file_path, "r") as txtfile:
        lines = txtfile.read().split('\n')
    return lines


def load_regular_coord_by_line(lines, line_id):
    line = lines[line_id]
    elems = line.split('\t')
    if len(elems) < 4:
        elems = line.split(',')
        if len(elems) < 4:
            elems = line.split(' ')

    try:
        [X1, Y1, W, H] = elems[0:4]
        coord_regular = [int(X1), int(Y1), int(W), int(H)]
        return coord_regular
    except IOError:
        print("Not enough ground truth in text file.")
        return False


def find_best_decimal_coord(multiple_coords_decimal, gt_coord_decimal):
    max_iou = 0
    for coord_decimal, id in enumerate(multiple_coords_decimal):
            iou = compute_iou_with_decimal_coord(coord_decimal, gt_coord_decimal)
            if iou >= max_iou:
                    max_iou = iou
                    index = id
    return multiple_coord_decimal[index]


def validate_coord(box):
    for i in range(len(box)):
        if math.isnan(box[i]):  box[i] = 0
