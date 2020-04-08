import os


def sampler(in_file='', out_dir=''):
    """Extracts grayscale i-frames from encoded video file.

    Keyword arguments:
    in_file -- input encoded video file (default '')
    out_dir -- the output directory for extracted frames (default '')
    """

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

    return './frames/sub'
