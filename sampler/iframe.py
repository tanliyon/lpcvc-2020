import getopt
import os
import sys
from math import sqrt


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

    # Extract I-frames and motion vectors
    os.system(f'ffmpeg -i {in_file} -f image2 -vf "select=\'eq(pict_type,PICT_TYPE_I)\'" -vsync vfr {out_dir} '
              f'-loglevel panic ')
    os.system(f'./extract_mvs {in_file} >/dev/null 2>&1')

    # Calculate motion scales per frame
    try:
        os.remove('motion_per_frame.txt')
    except OSError:
        pass
    frame_count = sum([len(files) for r, d, files in os.walk('./mv/')]) - 1
    for i in range(frame_count):
        os.system(f'awk \'{{print $1}}\' mv/frame{i+1}.txt | ./calculate_motion >> motion_per_frame.txt')
        os.system(f'awk \'{{print $2}}\' mv/frame{i+1}.txt | ./calculate_motion >> motion_per_frame.txt')

    # Create dictionary of motion scales per frame
    motion_per_frame = []
    with open('motion_per_frame.txt') as file:
        data = file.read().split()
    it = iter(data)
    for x, i in zip(it, range(frame_count)):
        displacement = (int(x) ** 2, int(next(it)) ** 2)
        motion_per_frame.append(sqrt(sum(displacement)))

    # Determine frames with low relative motion
    low_motion_frames = [1]
    for i in range(2, frame_count - 1):
        prev_scale = motion_per_frame[i - 1]
        if i == frame_count:
            if prev_scale > motion_per_frame[i]:
                low_motion_frames.append(i + 1)
        next_scale = motion_per_frame[i + 1]
        if prev_scale > motion_per_frame[i] and next_scale > motion_per_frame[i]:
            low_motion_frames.append(i + 1)
    return low_motion_frames


if __name__ == "__main__":
    frame_list = main(sys.argv[1:])
