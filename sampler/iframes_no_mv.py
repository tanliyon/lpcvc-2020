import os


def sampler(in_file='', out_dir=''):

    # Get program arguments
    if in_file == '':
        raise ValueError('No input file provided.')
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
    return sorted(os.listdir('../frames/'))
