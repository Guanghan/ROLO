# There are 3 kinds of representation of coordinates
# 1. Coord_decimal: (x0, y0, w, h) all represented in a float between [0, 1], ratio to image width and height, respectively.
#                   (x0, y0) is the middle point of the bounding box.
#                   Used by YOLO output, ROLO input and output.
# 2. Coord_regular: (X1, Y1, W, H) all represented by pixel values in int.
#                   (X1, Y1) is the top-left point of the bounding box
#                   Usually the ground truth box that is read from files are of this format
# 3. Detection in Vector: [4096-d feature_vector] + (class, x0, y0, w, h, prob).
#                   The same as Coord_decimal except that there are more information in the detection
#-----------------------------------------------------------------------------------------------

def coord_regular_to_decimal(coord_regular, img_wid, img_ht):
    img_wid *= 1.0
    img_ht *= 1.0
    coord_decimal = list(coord_regular)

    # convert top-left point (x,y) to mid point (x, y)
    coord_decimal[0] += coord_regular[2] / 2.0
    coord_decimal[1] += coord_regular[3] / 2.0

    # convert to [0, 1]
    coord_decimal[0] /= img_wid
    coord_decimal[1] /= img_ht
    coord_decimal[2] /= img_wid
    coord_decimal[3] /= img_ht

    return coord_decimal


def coord_decimal_to_regular(coord_decimal, img_wid, img_ht):
    w_box = int(coord_decimal[2] * img_wid)
    h_box = int(coord_decimal[3] * img_ht)
    x_topleft = int( img_wid * (coord_decimal[0] - coord_decimal[2]/2.0) )
    y_topleft = int( img_ht * (coord_decimal[1] - coord_decimal[3]/2.0) )

    coord_regular = [x_topleft, y_topleft, w_box, h_box]

    return coord_regular
