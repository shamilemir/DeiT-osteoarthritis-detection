note: ratios are off slightly due to "0" category being taken from two folders and "1" category from three folders
i.e. train/0 500, train/1 498, test/0 100, test/1 99


To generate a random selection of size 224x224 images:
-download original dataset from https://data.mendeley.com/datasets/56rmx5bjcr/1
-move contents of KneeOsteoarthritisRNN/ClsKLData/kneeKL224 into "all_data" (excluding "val" and "auto_test" folders)
-run the python scripts in terminal
  cd my/path/to/prepareKneeOsteoarthritisXray
  python ./datasplitting.py
  python ./preprocessing.py
- (optional) manually delete images with artifacts like implants

To generate a random selection of size 384x384 images:
-follow the above steps, but:
  -uncomment line 35 under # RESIZING in preprocessing.py
  -move contents of KneeOsteoarthritisRNN/ClsKLData/kneeKL299 instead of kneeKL224 (again excluding "val" and "auto_test" folders)


To redo selection process:
-replace selected_data and selected_processed_data with template_empty_folder and rename both
-repeat the other steps