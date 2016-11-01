import sys, os, shutil
sys.path.append(os.path.abspath("../utils/"))

def test_script(script_name):
    cmd = os.path.join(os.getcwd(), script_name)
    os.system('{} {}'.format('python', cmd))

def clean():
    shutil.rmtree('../temp_folder')

def main():
    scripts = ['test_utils_natural_sort.py',
               'test_utils_io_file.py',
               'test_utils_io_folder.py',
               'test_utils_io_coord.py',
               'test_utils_io_list.py',
               'test_utils_dataset.py',
               'test_utils_convert_coord.py']

    for script in scripts:
        test_script(script)

    clean()


if __name__ == '__main__':
    main()
