# LPCVC-2020 Sample Solution

## Overview
This is the sample solution for [low Power Computer Vision Challenge (LPCVC) 2020 Video Track](https://lpcv.ai/2020CVPR/video-track). This solution serves only as the baseline solution and a lot of improvements can be made on top of this to further optimize the performance of the solution.

The proposed solution is made up of 3 blocks. The first block (sampling block) takes in a video file and determine which frames are worth doing detection and recognition on. This sample solution does so by dissecting the motion vector from the H.264 encoding of the video to pick out stationary i-frames. The second block (detection block) does word recognition on the frames selected from the sampling block. This sample solution uses [EAST Detector](https://arxiv.org/abs/1704.03155). Lastly, the third block (recognition block) does optical character recognition (OCR) on the cropped words. The sample solution provides two choices: [Connectionist Temporal Classification (CTC)](https://arxiv.org/pdf/1507.05717.pdf) or [Attention OCR](https://arxiv.org/pdf/1704.03549.pdf).

## Contents
1. [Setup](#setup)
2. [Usage](#usage)
3. [Notes](#notes)

## Setup
1. Clone code from master branch.
  ```shell
  git clone https://github.com/tanliyon/lpcvc-2020.git
  ```
  
2. Download model file for all EAST-Detector, CTC and Attention OCR.\
  [EAST-Detector](https://drive.google.com/open?id=1g6mRhhrpOfCPrM9fEEmMS52IY72w8nbi) \
  [CTC](https://drive.google.com/open?id=1Hq484_MHM4wE7SY-d67HKbX52WbFD4To) \
  [Attention-Encoder](https://drive.google.com/open?id=1Z0suqT8qBZowBxIYncp5QWxTmQLoxrqf) \
  [Attention-Decoder](https://drive.google.com/open?id=1jiUDCuoqBYqD0460ozSVcxL_QwEc0Wua)
  
3. Install dependencies.\
  `pip install -r requirements.txt`\
  Note that lanms might not work with Windows.
  
4. Check directory structure. It should be:\
lpcvc-2020\
|\_wrapper.py\
|\_detector.pth\
|\_ctc.pth\
|\_encoder.pth\
|\_decoder.pth\
|\_(all other folders pulled from master)

## Usage
The call syntax is:
```shell
python main.py video_file_path.mp4 question_file_path.txt
```

To toggle between the two recognition option, you can toggle the `USE_ATTN_OCR` flag in main.py. The `SHOW_BOXES` flag controls if the detection output should be saved in a folder and the `SHOW_TEXT` flag controls if the recognition prediction should be printed in stdout.

## Notes
1. Currently, the solution takes a long time because of the number of frames it run inference on. If you want to test only a portion of it, run the code for a set amount of time, then comment out the line `frames_list = iFRAMES(video_path)` in wrapper.py. Then run the code again.
