import sys, os, io, shutil
sys.path.append(os.path.abspath("../utils/"))
from utils_io_folder import *

def test_create_folder():
    folder_path = "../temp_folder/"

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    create_folder(folder_path)

    if os.path.exists(folder_path):
        return True
    else:
        return False


def test_get_immediate_subfolder_paths():
    folder_path = '../temp_folder/'
    subfolder_paths = ['../temp_folder/subfolder_1/', '../temp_folder/subfolder_2']

    create_folder(folder_path)
    create_folder(subfolder_paths[0])
    create_folder(subfolder_paths[1])

    subfolder_paths_derived = get_immediate_subfolder_paths(folder_path)

    if set(subfolder_paths_derived).isdisjoint(subfolder_paths):
        return False
    else:
        return True


def test_get_immediate_subfolder_names():
    folder_path = '../temp_folder/'
    subfolder_paths = ['../temp_folder/subfolder_1/', '../temp_folder/subfolder_2']
    subfolder_names = ['subfolder_1', 'subfolder_2']

    create_folder(folder_path)
    create_folder(subfolder_paths[0])
    create_folder(subfolder_paths[1])

    subfolder_names_derived = get_immediate_subfolder_names(folder_path)

    if set(subfolder_names_derived).isdisjoint(subfolder_names):
        return False
    else:
        return True


def test_get_immediate_childfile_paths():
    temp_folder = '../temp_folder'
    create_dummy_files_in_folder(temp_folder)
    childfile_paths = [ os.path.join(temp_folder, (str(ct)+ '.txt')) for ct in range(10)]

    childfile_paths_derived = get_immediate_childfile_paths(temp_folder)
    shutil.rmtree(temp_folder)

    if set(childfile_paths_derived).isdisjoint(childfile_paths):
        return False
    else:
        return True


def test_get_immediate_childfile_names():
    temp_folder = '../temp_folder'
    create_dummy_files_in_folder(temp_folder)
    childfile_names = [(str(ct)+ '.txt') for ct in range(10)]

    childfile_names_derived = get_immediate_childfile_names(temp_folder)
    shutil.rmtree(temp_folder)

    if set(childfile_names_derived).isdisjoint(childfile_names):
        return False
    else:
        return True


def create_dummy_files_in_folder(temp_folder, file_format = 'txt'):
    create_folder(temp_folder)
    for ct in range(10):
        file_name = str(ct) + '.' + file_format
        file_path = os.path.join(temp_folder, file_name)
        with io.FileIO(file_path, "w") as file:
            file.write("Hello!")


def main():
    print("Testing: utils_io_folder")

    passed = test_create_folder()
    if passed is False:
        print("\t create_folder failed")

    passed = test_get_immediate_subfolder_names()
    if passed is False:
        print("\t get_immediate_subfolder_names failed")

    passed = test_get_immediate_subfolder_paths()
    if passed is False:
        print("\t get_immediate_childfile_paths failed")

    paseed = test_get_immediate_childfile_names()
    if passed is False:
        print("\t get immediate_childfile_names failed")

    passed = test_get_immediate_childfile_paths()
    if passed is False:
        print("\t get_immediate_childfile_paths failed")


if __name__ == '__main__':
    main()
