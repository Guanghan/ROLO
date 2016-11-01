import sys, os
sys.path.append(os.path.abspath("../utils/"))

from utils_io_coord import *
from utils_io_list import *
from utils_io_folder import create_folder

def test_batchload_yolovecs_from_numpy_folders():
    pairs_list_numpy_file_path = '/home/ngh/dev/ROLO-TRACK/training_list/list.npy'
    dataset_annotation_folder_path = '/home/ngh/dev/ROLO-dev/benchmark/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000'
    batchsize = 8
    nsteps = 3

    list_batch_pairs = load_list_batch_pairs_from_numpy_file(pairs_list_numpy_file_path, batchsize)
    for batch_pairs in list_batch_pairs[0:10]:
        batch_frame_ids = [batch_pair[1] for batch_pair in batch_pairs]
        batch_subfolder_names = [batch_pair[0] for batch_pair in batch_pairs]
        batch_numpy_folder_paths = [os.path.join(dataset_annotation_folder_path, subfolder_name, 'VID_loc_gt') for subfolder_name in batch_subfolder_names]

        attempted_yolo_batch = batchload_yolovecs_from_numpy_folders(batch_numpy_folder_paths, batch_frame_ids, batchsize, nsteps)
        if attempted_yolo_batch is not False:
            output_batch_vecs = attempted_yolo_batch
    return True


def test_save_vec_as_numpy():
    output_folder_path = '../temp_folder'
    create_folder(output_folder_path)

    pairs_list_numpy_file_path = '/home/ngh/dev/ROLO-TRACK/training_list/list.npy'
    dataset_annotation_folder_path = '/home/ngh/dev/ROLO-dev/benchmark/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000'
    batchsize = 8
    nsteps = 3

    list_batch_pairs = load_list_batch_pairs_from_numpy_file(pairs_list_numpy_file_path, batchsize)
    for batch_pairs in list_batch_pairs[0:10]:
        batch_frame_ids = [int(batch_pair[1]) for batch_pair in batch_pairs]
        batch_subfolder_names = [batch_pair[0] for batch_pair in batch_pairs]
        batch_numpy_folder_paths = [os.path.join(dataset_annotation_folder_path, subfolder_name, 'VID_loc_gt') for subfolder_name in batch_subfolder_names]

        attempted_yolo_batch = batchload_yolovecs_from_numpy_folders(batch_numpy_folder_paths, batch_frame_ids, batchsize, nsteps)
        if attempted_yolo_batch is not False:
            batch_output_vecs = attempted_yolo_batch
            for id in range(batchsize):
                frame_id = batch_frame_ids[id]
                output_vec = batch_output_vecs[id][nsteps-1]
                save_vec_as_numpy_by_frame_id(output_folder_path, frame_id, output_vec)
    return True


def main():
    print("Testing: utils_io_coord")

    finished = test_batchload_yolovecs_from_numpy_folders()
    if finished is not True:
        print("test_batchload_yolovecs_from_numpy_folder failed")

    finished = test_save_vec_as_numpy()
    if finished is not True:
        print("test_batchsave_vecs_as_numpy failed")


if __name__ == '__main__':
    main()
