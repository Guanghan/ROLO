import sys, os
sys.path.append(os.path.abspath("../utils/"))
from utils_io_file import *
from test_utils_io_folder import create_dummy_files_in_folder

def test_validate_file_format():
    temp_folder = '../temp_folder'
    create_dummy_files_in_folder(temp_folder, file_format = 'txt')
    create_dummy_files_in_folder(temp_folder, file_format = 'png')
    txt_file_path = os.path.join(temp_folder, '1.txt')
    png_file_path = os.path.join(temp_folder, '1.png')
    allowed_format = ['txt', 'jpg']

    expecting_true = validate_file_format(txt_file_path, allowed_format)
    expecting_false = validate_file_format(png_file_path, allowed_format)

    if expecting_true is True and expecting_false is False:
        return True
    else:
        return False


def main():
    print("Testing: utils_io_file")

    passed = test_validate_file_format()
    if passed is False:
        print("\t test_validate_file_format failed")


if __name__ == '__main__':
    main()
