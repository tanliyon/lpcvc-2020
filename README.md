# LPCVC-2020 Sample Solution

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
  
3. Download dependencies.\
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
