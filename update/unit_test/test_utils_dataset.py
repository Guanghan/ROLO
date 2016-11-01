import sys, os
sys.path.append(os.path.abspath("../utils/"))

from utils_io_coord import *
from utils_io_list import *
from utils_dataset import batchload_gt_decimal_coords_from_VID
from utils_io_folder import create_folder

def test_batchload_gt_decimal_coords_from_VID():
    VID_annotation_path = '/home/ngh/dev/ROLO-dev/benchmark/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000'
    pairs_list_numpy_file_path = '/home/ngh/dev/ROLO-TRACK/training_list/list.npy'
    batchsize = 8
    nsteps = 3
    offset = nsteps - 1

    list_batch_pairs = load_list_batch_pairs_from_numpy_file(pairs_list_numpy_file_path, batchsize)

    for batch_pairs in list_batch_pairs[0:2]:
        batch_subfolder_names = [batch_pair[0] for batch_pair in batch_pairs]
        batch_frame_ids = [int(batch_pair[1]) for batch_pair in batch_pairs]

        batch_gt_decimal_coords = batchload_gt_decimal_coords_from_VID(VID_annotation_path, batch_subfolder_names, batch_frame_ids, offset)

        if batch_gt_decimal_coords is False:
            return False
        else:
            return True


def main():
    print("Testing: utils_io_dataset")

    finished = test_batchload_gt_decimal_coords_from_VID()
    if finished is not True:
        print("test_batchload_gt_decimal_coords failed")


if __name__ == '__main__':
    main()
