#### To generate a random selection of size 224x224 images:
1. Download original dataset from https://data.mendeley.com/datasets/56rmx5bjcr/1
2. Create folder named **all_data**
3. Copy and paste **template_empty_folder** twice and rename to **selected\_data** and **selected\_processed\_data**
4. Move contents of **KneeOsteoarthritisRNN/ClsKLData/kneeKL224** into **all\_data** (excluding "val" and auto\_test folders)
5. run the python scripts in terminal
    - `cd my/path/to/prepareKneeOsteoarthritisXray`
    - `python ./datasplitting.py`
    - `python ./preprocessing.py`
6.  (optional) manually delete images with artifacts like implants
7.  (optional) color invert images that are randomly color inverted the wrong way


#### To generate a random selection of size 384x384 images:
1. follow the above steps, but:
    - uncomment line 35 under # RESIZING in preprocessing.py
    - move contents of **KneeOsteoarthritisRNN/ClsKLData/kneeKL299** instead of **kneeKL224** (again excluding "val" and "auto\_test" folders)


#### To redo selection process:
1. replace **selected\_data** and **selected\_processed\_data** with **template\_empty\_folder** and rename both again
2. repeat remaining steps


*Note: ratios are off slightly due to "0" category being taken from two folders and "1" category from three folders*
- train/0: 500
- train/1: 498
- test/0: 100
- test/1: 99
