import sys, os
sys.path.append(os.path.abspath("../utils/"))
from utils_io_list import *
from test_utils_io_folder import *

def test_generate_pairs_for_each_folder():
    images_folder_path= "folder/path/example"
    num_of_frames = 2

    pairs = generate_pairs_for_each_folder(images_folder_path, num_of_frames)

    expected_pair = [("example", 0), ("example", 1)]
    if expected_pair == pairs:
        return True
    else:
        return False


def test_generate_num_of_frames_list():
    folders_paths_list = ['../temp_folder_1', '../temp_folder_2']
    for folder_path in folders_paths_list:
        create_folder(folder_path)
        create_dummy_files_in_folder(folder_path)

    num_of_frames_list = generate_num_of_frames_list(folders_paths_list)

    for folder_path in folders_paths_list:
        shutil.rmtree(folder_path)

    expected_list = [10, 10]
    if expected_list == num_of_frames_list:
        return True
    else:
        return False


def test_generate_pairs_with_two_lists():
    folders_paths_list = ['../temp_folder_1', '../temp_folder_2']
    num_of_frames_list = [1, 2]

    pairs_list = generate_pairs_with_two_lists(folders_paths_list, num_of_frames_list)

    expected_list = [('temp_folder_1', 0), ('temp_folder_2', 0), ('temp_folder_2', 1)]
    if expected_list == pairs_list:
        return True
    else:
        return False


def test_generate_pairs_list_for_training():
    dataset_folder_path = '/home/ngh/dev/ROLO-dev/benchmark/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/'
    output_folder_path = '/home/ngh/dev/ROLO-TRACK/training_list/'
    create_folder(output_folder_path)

    txt_file_path = os.path.join(output_folder_path, 'list_0.txt')
    numpy_file_path = os.path.join(output_folder_path, 'list_0')

    finished = generate_pairs_list_for_training(dataset_folder_path, numpy_file_path, txt_file_path)

    if finished is True:
        return True
    else:
        return False


def main():
    print("Testing: utils_io_list")

    passed = test_generate_num_of_frames_list()
    if passed is False:
        print("test_generate_num_of_frames_list failed")

    passed = test_generate_pairs_for_each_folder()
    if passed is False:
        print("test_generate_pairs_for_each_folder failed")

    passed = test_generate_pairs_with_two_lists()
    if passed is False:
        print("test_generate_pairs_with_two_lists failed")

    passed = test_generate_pairs_list_for_training()
    if passed is False:
        print("test_generate_pairs_list_for_training failed")


if __name__ == "__main__":
    main()
