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
        out_dir = 'frames/frame%03d.png'
    else:
        out_dir = out_dir + 'frame%03d.png'

    # Extract I-frames
    os.system(f'ffmpeg -i {in_file} -f image2 -vf "select=\'eq(pict_type,PICT_TYPE_I)\'" -vsync vfr {out_dir} '
              f'-hide_banner -loglevel panic')

    # Extract motion vectors
    os.system(f'./extract_mvs {in_file}')

    # Calculate motion per frame
    frame_count = sum([len(files) for r, d, files in os.walk('./mv/')]) - 1
    motion_per_frame = {}
    for i in range(frame_count):
        print(f'Frame {i+1} motion values:')
        os.system(f'awk \'{{print $1}}\' mv/frame{i+1}.txt | ./calculate_motion')
        os.system(f'awk \'{{print $2}}\' mv/frame{i+1}.txt | ./calculate_motion')
        print('')


if __name__ == "__main__":
    main(sys.argv[1:])
