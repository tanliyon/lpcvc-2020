# LPCVC-2020 Sample Solution

## Overview
This is the sample solution for [low Power Computer Vision Challenge (LPCVC) 2020 Video Track](https://lpcv.ai/2020CVPR/video-track). This solution serves only as the baseline solution and a lot of improvements can be made on top of this to further optimize the performance of the solution.

The proposed solution is made up of 3 blocks. The first block (sampling block) takes in a video file and determine which frames are worth doing detection and recognition on. This sample solution does so by dissecting the motion vector from the H.264 encoding of the video to pick out stationary i-frames. The second block (detection block) does word recognition on the frames selected from the sampling block. This sample solution uses [EAST Detector](https://arxiv.org/abs/1704.03155). Lastly, the third block (recognition block) does optical character recognition (OCR) on the cropped words. The sample solution uses [Connectionist Temporal Classification (CTC)](https://arxiv.org/pdf/1507.05717.pdf) for now.

## Contents
1. [Setup](#setup)
2. [Usage](#usage)
3. [Notes](#notes)

## Setup
1. Clone code from master branch.
  ```shell
  git clone https://github.com/tanliyon/lpcvc-2020.git
  ```
  
2. Download model file for both EAST-Detector and CTC.\
  EAST-Detector: Not ready\
  CTC: https://drive.google.com/open?id=1Hq484_MHM4wE7SY-d67HKbX52WbFD4To
  
3. Install dependencies.\
  i. lanms - `pip install lanms`
  
4. Check directory structure. It should be:\
lpcvc-2020\
|\_wrapper.py\
|\_detector.pth (not ready)\
|\_ctc.pth\
|\_(all other folders pulled from master)

## Usage
The call syntax is:
```shell
python wrapper.py video_file_path.MP4 question_file_path.txt answer_file_path.txt
```

## Notes
1. Currently, the first block of the code is taking really long. If you want to test only a portion of it, run the code for a set amount of time, then comment out the line `frames_list = iFRAMES(video_path)` in wrapper.py. Then run the code again.
2. Since the current detector is not trained, the output will rarely match any questions. Hence, the answer file will most likely be empty.
3. Currently, the recognition block uses CTC to do OCR. We are experimenting with using attention mechanism and that will most likely be the one we use instead since the performance is probably better.
4. The current solution is not tested on the Raspberry Pi 3b+ yet. Hence, we are not sure if it will run or not on the pi.
