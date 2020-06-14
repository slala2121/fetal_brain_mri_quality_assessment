# fetal_image_quality_assessment

Author: Sayeri Lala
email: ls2121@mit.edu, lsls21079@gmail.com
date: May 2019

Code for experiments reported in:
  -Lala, et al. A Deep Learning Approach for Image Quality Assessment of Fetal Brain MRI. ISMRM 2019.
  -thesis: Sayeri Lala, "Convolutional Neural Networks for 2D Slice Image Reconstruction and Image Quality Assessment of Fetal Brain MRI". Massachusetts Institute of Technology, 2019.

  -any code used from another source was attributed in the source files



Citation and acknowledgment
-------------------------------------------------------------
If you find the code/data/literature helpful please cite the following works:

-Lala, et al. A Deep Learning Approach for Image Quality Assessment of Fetal Brain MRI. ISMRM 2019.
-thesis: Sayeri Lala, "Convolutional Neural Networks for 2D Slice Image Reconstruction and Image Quality Assessment of Fetal Brain MRI". Massachusetts Institute of Technology, 2019.


Software
-------------------------------------------------------------
1. Matlab

2. Python (tested on version 2)
  -tensorflow-gpu
  -keras 
  -numpy
  -matplotlib
  -scipy
  -nibabel
  -keras-vis  
      -MUST MAKE THE following change to vis/visualization/... file: 
      overwrite the visualization/saliency.py:: visualize_cam_with_losses with the function definition provided in 'saliency_edits.py'
  -PIL
  -csv
  -pydicom

3. RStudio
  -ROCR
  -binom
  -pROC
  -boot


Data and code for the experiments
-------------------------------------------------------------
Labeled data and train/val/test partitions prepared for the experiments available under:
gpu.mit.edu: ~/../../unborn/shared/sayeri/iqa_data_source


Code: fork the 'cleaned' branch under the image_quality repository. 


Training, evaluation scripts
-------------------------------------------------------------
Creating the training/validation/test sets:
1. run create_train_val_test.py

Training/evaluation (ROC, saliency maps)
1. Train and evaluate models: run call_run_pipeline.sh i.e., in the terminal ./call_run_pipeline.sh

    Note that filepaths in run_pipeline.sh need to be updated.

2. Generate ROCs + statistical tests: run ci_metrics_final.R

More details available in these source files.


Testing:
1. test_data for running tests in test/testDataset.py can be found under:
gpu.mit.edu: ~/../../unborn/shared/sayeri/test_data


Backup
-------------------------------------------------------------

5-28-19. This code has been backed up on gpu.mit.edu: ~/../../unborn/shared/sayeri/image_quality


Dataset
-------------------------------------------------------------

Labeled data and train/val/test partitions prepared for the experiments available under:
gpu.mit.edu: ~/../../unborn/shared/sayeri/iqa_data_source



Preparing and labeling NEW data:
1. download the data source folder
2. run saveHASTEImages.m to reorganize and prepare the data for labeling
3. for each dicom folder:
	-use the LabelBox interface to label data (below)
	-save the *.csv to the stack_# folder 



Analyzing data:
1. run analyze_data.py to analyze quality distribution
2. characterizeDataset.m to analyze acquisition params




LabelBox instructions:

Labeling Interface (as of 4-24-19)
1. Image Classification
2. For the labeling interface, use the following template
3. Labeling process for each stack. Go to Settings.
    1. Click Datasets. Under Existing Data, detach any previous stack data. 
    2. Click Upload New Data. Upload the stack data
    3. Refresh. Click Start Labeling.
    4. After finishing the labeling, click Export.
            ***Note it could take time for all the labels to be uploaded so you might need to wait and hit refresh several times**** 
    5. Set Export format: CSV. Click Generate Export. 
      Download labels file to the stack directory.
    6. Under Labels, click Delete & Reenque to delete the old stack labels. 


Labeling interface template
[
  {
    "name": "image_quality",
    "instructions": "Select good/bad/uncertain",
    "type": "radio",
    "required": false,
    "options": [
      {
        "value": "good",
        "label": "good"
      },
      {
        "value": "bad",
        "label": "bad"
      },
      {
      	"value": "uncertain",
        "label": "uncertain"
      }
    ]
  },
  {
    "name": "roi",
    "instructions": "roi present yes/no",
    "type": "radio",
    "required": false,
    "options": [
      {
        "value": "yes",
        "label": "yes"
      },
      {
        "value": "no",
        "label": "no"
      }
    ]
  }
]










