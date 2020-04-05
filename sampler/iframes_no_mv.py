import os


def sampler(in_file='', out_dir=''):

    # Get program arguments
    if in_file == '':
        raise ValueError('No input file provided.')
    if out_dir == '':
        out_dir = './frames/sub/frame%03d.jpg'
    else:
        out_dir = out_dir + 'frame%03d.jpg'

    try:
        os.mkdir('./frames/sub')
    except FileExistsError:
        pass

    # Extract I-frames and motion vectors
    os.system(f'ffmpeg -i {in_file} -vf select="eq(pict_type\,I)" -an -vsync 0 {out_dir} -loglevel panic 2>&1')
    os.system(f'ffmpeg -i {out_dir} -vf format=gray {out_dir} -loglevel panic')

    # Determine frames with low relative motion
    try:
        os.remove('frames/.DS_Store')
    except OSError:
        pass
    return './frames/sub'


if __name__ == '__main__':
    sampler(in_file='../videos/short.mp4')