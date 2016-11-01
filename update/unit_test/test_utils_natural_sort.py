import sys, os
sys.path.append(os.path.abspath("../utils/"))
from utils_natural_sort import *

def test_natural_sort():
    test_string_list_desired = ['A00001', 'A00002', 'A00010', 'A00011', 'B00001', 'B00002', 'B00010', 'B00011']
    test_string_list = ['B00002', 'A00010', 'A00011', 'B00010', 'A00001', 'B00011', 'A00002', 'B00001']

    natural_sort(test_string_list)

    if test_string_list == test_string_list_desired:
        return True
    else:
        return False


def main():
    print("Testing: utils_natural_sort")

    passed = test_natural_sort()
    if passed is False:
        print("\t natural_sort failed")


if __name__ == '__main__':
    main()
