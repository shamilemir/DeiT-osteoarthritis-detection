You can find the corresponding journal articel at https://www.urncst.com/index.php/urncst/article/view/789

#### To generate a random selection of size 224x224 images:
1. Download original dataset from https://data.mendeley.com/datasets/56rmx5bjcr/1
*Complete the following steps within the prepareKneeOsteoarthritisXray folder*
2. Create folder named **all\_data**
3. Create a **template\_empty\_folder** with two subfolders **train** and **test** and each of those with two subfolders **0** and **1**
4. Copy and paste **template\_empty\_folder** twice and rename to **selected\_data** and **selected\_processed\_data**
5. Move contents of **KneeOsteoarthritisRNN/ClsKLData/kneeKL224** into **all\_data** (excluding "val" and auto\_test folders)
6. Run the python scripts in terminal
    - `cd my/path/to/prepareKneeOsteoarthritisXray`
    - `python ./datasplitting.py`
    - `python ./preprocessing.py`
7.  (Optional) manually delete images with artifacts like implants
8.  (Optional) color invert images that are randomly color inverted the wrong way


#### To generate a random selection of size 384x384 images:
1. Follow the above steps, but in addition:
    - Uncomment line 35 under "RESIZING" in preprocessing.py
    - Move contents of **KneeOsteoarthritisRNN/ClsKLData/kneeKL299** instead of **kneeKL224** (again excluding "val" and "auto\_test" folders)


#### To redo selection process:
1. Replace **selected\_data** and **selected\_processed\_data** with **template\_empty\_folder** and rename both again
2. Repeat remaining steps


*Note: ratios are off slightly due to "0" category being taken from two folders and "1" category from three folders*
- train/0: 500
- train/1: 498
- test/0: 100
- test/1: 99

#### You are now ready to fine-tune using the fine-tune.py script