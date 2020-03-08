import getopt
import os
import sys


def main(argv):
    in_file = ''
    out_dir = ''

    # Get program arguments
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        raise ValueError('usage: python3 iframe.py -i <infile> [-o <outdir>]')
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            in_file = arg
        elif opt in ("-o", "--ofile"):
            out_dir = arg

    if in_file == '':
        raise ValueError('usage: python3 iframe.py -i <inputfile>')

    if out_dir == '':
        out_dir = '../frames/frame%03d.png'
    else:
        out_dir = out_dir + 'frame%03d.png'

    try:
        os.mkdir('../frames')
    except FileExistsError:
        pass

    # Extract I-frames and motion vectors
    os.system(f'ffmpeg -i {in_file} -f image2 -vf "select=\'eq(pict_type,PICT_TYPE_I)\'" -vsync vfr {out_dir} '
              f'-loglevel panic ')

    # Determine frames with low relative motion
    try:
        os.remove('frames/.DS_Store')
    except OSError:
        pass
    return list(range(1, len(os.listdir('frames/')) + 1))


if __name__ == "__main__":
    frame_list = main(sys.argv[1:])
    print(frame_list)
