# Data for the model
## Custom Dataset Class
- TextLocalizationSet
- WordRecognitionSet
- Focused Scene Text
- COCO Text Words
### Text Localization Set
- The goal of this dataset is to train the model to locate text in the images
- This custom Dataset class is written with ICDAR 2015 Incidental Scene Text (https://rrc.cvc.uab.es/?ch=4)
- Train: 1000 images with 1000 corresponding ground truth
- Test: 500 images with 500 corresponding ground truth
- Ground truth contains bounding box coordinates and transcription
- Illegible texts are marked with "###" in the ground truth file and removed with script under 'invalid_data_filter' directory
- **__NOTE: UPDATE VERSION OF THIS CLASS IS UNDER EAST DETECTOR BRANCH__**
#### invalid_data_filter
- Feed in image and ground truth directories for Text Localization Set to remove invalid transcriptions
- usage: python dataset_filter.py <path_to_image_folder> <path_to_gt_folder>
### Word Recognition Set
- The goal of this dataset is to train the model to recognize words in cropped out text images
- This custom Dataset class is wirtten with ICDAR 2015 Incidental Scene Text (https://rrc.cvc.uab.es/?ch=4)
- Train: 4468 images with transcriptions (ground truth)
- Test: 2077 images with transcriptions (ground truth)
- Ground truth contains actual word transcriptions
### Focused Scene Text
- This is also an ICDAR dataset (https://rrc.cvc.uab.es/?ch=2)
- Simply manipulate/reformat this dataset to make it compatible with TextLocalization and WordRecognition custom Dataset class above
#### For Text Localization Set:
##### convert_dataset_gt.py
- This script will identically format the ground truth text files with Incidental Scene Text dataset 
- usage: python convert_dataset_gt.py <path to ground truth directory>
##### convert_file_name.py
- This script will rename the file names to allow us to merge the files in one directory (make them contiguous as if they were one dataset to begin with)
- Have to manually change the file name string for image files and ground truth files
- Usage: python convert_file_name.py <path/directory to the files> <starting label number (ex. img_{label_number}.png)>
#### For Word Recognition Set:
- This script will identically format the image and ground truth text files with Incidental Scene Text dataset
- usage: python convert_word_recognition.py <mode: -f for file name change, -c for ground truth content change> <directory name> <starting number label>

- After running these scripts, manually copy/cut paste the reformatted files to original base dataset directory for merging :(
### COCO Text Words Dataset
- Almost identical to the Word Recognition Set of ICDAR 2015 but larger and more balanced
- Train: 42,618 images with ground truth
- Val: 9,896 images with ground truth
- test: 9,837 images but no ground truth (no good so don't use test set)
