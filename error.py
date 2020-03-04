import os
import sys
import glob

def does_directory_exist(path):
    if not os.path.isdir(path):
        raise OSError("Directory path {} provided does not exist".format(path))

def does_file_exist(path):
    if not os.path.isfile(path):
        raise OSError("File path {} provided does not exist".format(path))
