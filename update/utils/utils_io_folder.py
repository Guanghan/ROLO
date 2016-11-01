import os
from utils_natural_sort import natural_sort

def get_immediate_subfolder_paths(folder_path):
    subfolder_names = get_immediate_subfolder_names(folder_path)
    subfolder_paths = [os.path.join(folder_path, subfolder_name) for subfolder_name in subfolder_names]
    return subfolder_paths


def get_immediate_subfolder_names(folder_path):
    subfolder_names = [folder_name for folder_name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, folder_name))]
    natural_sort(subfolder_names)
    return subfolder_names


def get_immediate_childfile_paths(folder_path):
    files_names = get_immediate_childfile_names(folder_path)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def get_immediate_childfile_names(folder_path):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    natural_sort(files_names)
    return files_names


def get_folder_name_from_path(folder_path):
    path, folder_name = os.path.split(folder_path)
    return folder_name


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
