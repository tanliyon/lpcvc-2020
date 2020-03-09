#  I-frame Interval Sampling

```bash
./compile
mkdir mv

# entire pipeline, command line
python3 iframe.py -i <infile>

# just iframes, import iframes_no_mv.py
frames = sampler(infile)

# just motion vectors
./extract_mv <infile>
```
