#  I-frame Interval Sampling

To extract I-frames from a .mp4 video file, run the following command:
```
python3 iframe.py -i <infile> [-o <outdir>]`
```
The `-o <outdir>` argument is optional. By default, the I-frames will be output to a `frames/` directory. Motion vectors for each frame will be extracted to a `mv/` directory.

The format for the motion vector files is `motion_x, motion_y, dx, dy` where the first 2 entries are the motion scales in the x and y axes and the latter 2 entries are the change in position of the center of that entry's macroblock.

Currently, the summed motion value for each frame is outputted to `stdout`.
