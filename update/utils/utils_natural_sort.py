import re

def natural_sort(given_list):
    """ Sort the given list in the way that humans expect."""
    given_list.sort(key=alphanum_key)


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"] """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def tryint(s):
    try:
        return int(s)
    except:
        return s
