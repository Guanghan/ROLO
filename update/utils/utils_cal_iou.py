
def compute_iou_with_regular_coord(box1, box2):
    # Prevent NaN in benchmark results
    validate_box(box1)
    validate_box(box2)

    # change float to int, in order to prevent overflow
    box1 = map(int, box1)
    box2 = map(int, box2)

    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb <= 0 or lr <= 0 :
        intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


def compute_iou_with_decimal_coord(box1, box2, w, h):
    box1 = coord_decimal_to_regular(w,h,box1)
    box2 = coord_decimal_to_regular(w,h,box2)
    return compute_iou_with_regular_coord(box1,box2)


def cal_score(location, gt_location, thresh):
    iou_score = compute_iou_with_regular_coord(regular_box1, regular_box2)
    if iou_score >= thresh:
        score = 1
    else:
        score = 0
    return score
