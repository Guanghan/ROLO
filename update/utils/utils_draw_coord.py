from utils_convert_coord import coord_regular_to_decimal, coord_decimal_to_regular
import cv2

def debug_decimal_coord(img, coord_decimal, prob = None, class_id = None):
    img_cp = img.copy()
    img_ht, img_wid, nchannels = img.shape

    coord_regular = coord_decimal_to_regular(coord_decimal, img_wid, img_ht)

    debug_regular_coord(img, coord_regular, prob, class_id)


def debug_regular_coord(img, coord_regular, prob = None, class_id = None):
    img_cp = img.copy()
    [x_topleft, y_topleft, w_box, h_box] = coord_regular

    cv2.rectangle(img_cp,
                 (x_topleft, y_topleft),
                 (x_topleft + w_box, y_topleft + h_box),
                 (0,255,0), 2)

    if prob is not None and class_id is not None:
        assert(isinstance(prob, (float)))
        assert(isinstance(class_id, (int, long)))
        cv2.rectangle(img_cp,
                      (x_topleft, y_topleft - 20),
                      (x_topleft + w_box, y_topleft),
                      (125,125,125),-1)
        cv2.putText(img_cp,
                    str(class_id) + ' : %.2f' % prob,
                    (x_topleft + 5, y_topleft - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.imshow('debug_detection',img_cp)
    cv2.waitKey(1)


def debug_3_locations( img, gt_location, yolo_location, rolo_location):
    img_cp = img.copy()
    for i in range(3):  # b-g-r channels
        if i== 0: location= gt_location; color= (0, 0, 255)       # red for gt
        elif i ==1: location= yolo_location; color= (255, 0, 0)   # blur for yolo
        elif i ==2: location= rolo_location; color= (0, 255, 0)   # green for rolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 1 or i== 2: cv2.rectangle(img_cp,(x-w//2, y-h//2),(x+w//2,y+h//2), color, 2)
        elif i== 0: cv2.rectangle(img_cp,(x,y),(x+w,y+h), color, 2)
    cv2.imshow('3 locations',img_cp)
    cv2.waitKey(100)
    return img_cp
