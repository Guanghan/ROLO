import os

def validate_file_format(file_in_path, allowed_format):
    if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in allowed_format:
        return True
    else:
        return False


class Error(Exception):
    """Base class for other exceptions"""
    pass


class FormatIncorrectError(Error):
    """Raised when the file is of incorrect format"""
    pass


def is_image(file_in_path):
    if validate_file_format(file_in_path, ['jpg', 'JPEG', 'png', 'JPG']):
        return True
    else:
        return False


def is_video(file_in_path):
    if validate_file_format(file_in_path, ['avi', 'mkv', 'mp4']):
        return True
    else:
        return False


def file_to_img(filepath):
    try:
        img = cv2.imread(filepath)
    except IOError:
        print('cannot open image file: ' + filepath)
    else:
        print('unknown error reading image file')
    return img


def file_to_video(filepath):
    try:
            video = cv2.VideoCapture(filepath)
    except IOError:
            print('cannot open video file: ' + filepath)
    else:
            print('unknown error reading video file')
    return video
