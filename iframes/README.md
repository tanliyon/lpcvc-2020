#  I-frame Interval Sampling

To extract I-frames from a .mp4 video file, run the following command:
```
python3 iframe.py -i <infile> [-o <outdir>]`.
```
By default, the I-frames will be output to a `frames/` directory. The output argument is optional.

To extract motion vectors, run 
```
./compile
extract_mvs <infile>
```
The motion vectors will be output to the `output/mv/` directory.
