import numpy as np
from random import shuffle
from utils_io_folder import get_immediate_subfolder_paths, get_immediate_childfile_names, get_folder_name_from_path

''' 1. generate the list of pairs '''

def generate_pairs_list_for_training(dataset_folder_path, numpy_file_path, txt_file_path = None):
    folders_paths_list = generate_folders_paths_list(dataset_folder_path)
    num_of_frames_list = generate_num_of_frames_list(folders_paths_list)

    pairs_list = generate_pairs_with_two_lists(folders_paths_list, num_of_frames_list)
    shuffled_pairs_list= shuffle_list(pairs_list)

    generate_numpy_file_with_shuffled_list_of_pairs(numpy_file_path, shuffled_pairs_list)
    if txt_file_path is not None:
        generate_txt_file_with_shuffled_list_of_pairs(txt_file_path, shuffled_pairs_list)

    return True


def generate_folders_paths_list(dataset_folder_path):
    folders_paths_list = get_immediate_subfolder_paths(dataset_folder_path)
    return folders_paths_list


def generate_num_of_frames_list(folders_paths_list):
    num_of_frames_list = [len(get_immediate_childfile_names(folder_path))
                         for folder_path in folders_paths_list]
    return num_of_frames_list


def generate_pairs_with_two_lists(folders_paths_list, num_of_frames_list):
    pairs_list = []
    assert(len(folders_paths_list) == len(num_of_frames_list))

    for folder_id, images_folder_path in enumerate(folders_paths_list):
        num_of_frames = num_of_frames_list[folder_id]
        pairs = generate_pairs_for_each_folder(images_folder_path, num_of_frames)

        for pair in pairs:
            pairs_list.append(pair)
    return pairs_list


def generate_pairs_for_each_folder(images_folder_path, num_of_frames):
    pairs =[(get_folder_name_from_path(images_folder_path), ct)
            for ct in range(num_of_frames)]
    return pairs


def generate_txt_file_with_shuffled_list_of_pairs(txt_file_path, shuffled_pairs_list):
    try:
        with open(txt_file_path, "w") as txt_file:
            for pairs in shuffled_pairs_list:
                line = str(pairs) + '\n'
                txt_file.write(line)
    except IOError:
        print('unable to open text file')


def generate_numpy_file_with_shuffled_list_of_pairs(numpy_file_path, shuffled_pairs_list):
    np.save(numpy_file_path, shuffled_pairs_list)


def shuffle_list(pairs_list):
    shuffle(pairs_list)
    return pairs_list


''' 2. Load the list of pairs '''
def load_list_batch_pairs_from_numpy_file(pairs_list_numpy_file_path, batchsize):
    shuffled_pairs_list = load_pairs_list_from_numpy_file(pairs_list_numpy_file_path)
    list_batch_pairs = convert_pairs_to_list_batch_pairs(shuffled_pairs_list, batchsize)
    return list_batch_pairs


def load_pairs_list_from_numpy_file(pairs_list_numpy_file_path):
    shuffled_pairs_list = np.load(pairs_list_numpy_file_path)
    return shuffled_pairs_list


def convert_pairs_to_list_batch_pairs(pairs_list, batchsize):
    list_batch_pairs = []
    for batch_id in range(0, len(pairs_list), batchsize):
        st = batch_id
        ed = st + batchsize
        batch_pairs = pairs_list[st:ed]
        list_batch_pairs.append(batch_pairs)
    return list_batch_pairs
