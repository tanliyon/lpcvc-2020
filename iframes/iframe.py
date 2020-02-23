import getopt
import os
import sys


def main(argv):
    inFile = ''
    outDir = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        raise ValueError('usage: python iframe.py -i <infile> [-o <outdir>]')
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inFile = arg
        elif opt in ("-o", "--ofile"):
            outDir = arg

    if inFile == '':
        raise ValueError('usage: python iframe.py -i <inputfile>')

    if outDir == '':
        outDir = 'frames/frame%03d.png'
    else:
        outDir = outDir + 'frame%03d.png'

    os.system(f'ffmpeg -i {inFile} -f image2 -vf "select=\'eq(pict_type,PICT_TYPE_I)\'" -vsync vfr {outDir}')


if __name__ == "__main__":
    main(sys.argv[1:])
