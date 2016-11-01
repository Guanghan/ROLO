import sys, os
sys.path.append(os.path.abspath("../utils/"))
from utils_io_file import *
from test_utils_io_folder import create_dummy_files_in_folder
from utils_convert_coord import *
import numpy as np

def test_coord_decimal_to_regular():
    [img_wid, img_ht] = [640, 480]
    coord_decimal = [0.44312766, 0.64272517, 0.15378259, 0.27607924]

    coord_regular_converted = coord_decimal_to_regular(coord_decimal, img_wid, img_ht)
    coord_decimal_converted = coord_regular_to_decimal(coord_regular_converted, img_wid, img_ht)

    print("\t decimal coords                 : " + str(coord_decimal))
    print("\t decimal coords after conversion: " + str(coord_decimal_converted))

    loss = sum(abs(np.array(coord_decimal_converted) - np.array(coord_decimal)))

    if loss <= 0.004:
        return True
    else:
        print("loss for decimal coords is: " + str(loss))
        return False


def test_coord_regular_to_decimal():
    [img_wid, img_ht] = [640, 480]
    coord_regular = [234, 242, 98, 132]

    coord_decimal_converted = coord_regular_to_decimal(coord_regular, img_wid, img_ht)
    coord_regular_converted = coord_decimal_to_regular(coord_decimal_converted, img_wid, img_ht)

    print("\t regular coords                 : " + str(coord_regular))
    print("\t regular coords after conversion: " + str(coord_regular_converted))

    loss = sum(abs(np.array(coord_regular_converted) - np.array(coord_regular)))
    if loss <= 4:
        return True
    else:
        print("loss for regular coordinates is: " + str(loss))
        return False


def main():
    print("Testing: utils_convert_coord")

    passed = test_coord_decimal_to_regular()
    if passed is False:
        print("\t test_coord_decimal_to_regular failed")

    passed = test_coord_regular_to_decimal()
    if passed is False:
        print("\t test_coord_regular_to_decimal failed")

if __name__ == '__main__':
    main()
