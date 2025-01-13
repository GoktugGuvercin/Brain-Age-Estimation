# Brain Age Estimation

## Background Information

Brain diseases like alzheimer, tramumatic injury, and psychiatric disorders (schizophrenia: reduced gray matter, depression: alteration in the hippocampus) may pose some structural changes or potential damage on the brain tissue. At this point, neurodegenerative diseases comes into the forefront. In this case, predicting the age of a patient from their brain MRI scan can be used as a promising biomarker for early and subtle brain changes.

The difference between predicted brain age and actual chronological age is called as ***brain age gap***. When predicted value is significantly larger than real body age, it is concluded that brain characteristics of that person are older than his/her biology, which is called ***accelerated aging***. In that case, that person is expected to go through one of those brain diseases. In contrast, certain lifestyle factors or protective genetic traits can slow down brain aging and even keep it lower than chronological age.

## Purpose

The objective for this project is to design regression models to be able to predict brain age from MRI scans. It is based on supervised learning, and it requires healthy reference subjects, as provided in the dataset. This project was prepared by AI in Medicine Lab in Technical University of Munich, and preliminary source files are accessible publicly in their github repository: https://github.com/compai-lab/aim-practical-1-brain-age-estimation


## Dataset
The dataset comprises T1-Weighted MRI scans of 652 healthy subjects. It is partitioned into 3 folds for training, validation and testing stages.

- Train: 500 images
- Validation: 47 images
- Test: 105 images

Each image is represented in NIFTI format. Together with MRI scans, a brain region mask and segmentation mask are provided. Original dataset is kept as private.