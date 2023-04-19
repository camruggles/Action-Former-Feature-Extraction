
## Step 1 : extract features for training set

First, a baseline set of features must be extracted using i3d backbone.

* go to extract.py in scripts
* changes lines 12-16 accordingly
* run these lines

```shell
ls "folder of completed features"   > list.txt
ls "folder of validation files"     > todo.txt
```
then

```shell
python extract.py
```

this will compute features for all the validation videos using the i3d backbone




## Step 2: extract time stamps

To speed up video loading during training, it is necessary to extract video meta data.

in stamps.py (in scripts)
* change line 14 to an output folder
* on line 17 have a filename that contains the filenames to look at

run the file stamps.py
```shell
python stamps.py
```

this will extract and save all of the time stamps for the validation videos in a stamps folder






## Step 3 : combine extracted features with optical flow features
Here it is necessary to download the features from the actionformer repository extracted with the FINSPIRE repo in order to get the flow features.
Come back here when you have downloaded and extracted the i3d features from the original actionformer repo.


run

```shell
ls " folder where validation FEATURES are saved"     > transfer.txt
```

* change lines 8-10 in the setfeatures.py to reflect the correct file paths

use setfeatures.py to save the newly extracted features over the old files, where the optical flow data is saved





## Step 4 : run training

set the parameters in the config yaml and then run training

```shell
python ./train.py ./configs/thumos_i3d.yaml --output "name of the checkpoint file you want"
```



## Step 5 : extract test features

The network from step 4 will be saved somewhere as dictated by the yaml
use this network to re extract new features on the test videos

First, repeat step 2 on the test videos.  On a multicore computer step 2 should only take 5 minutes.

```shell
python testextraction.py
```

*edit the file paths starting at line 12 of testextraction.py and then
*do  these commands
```shell
ls "folder of test features" > list.txt
ls "folder of test videos" > todo.txt
python testextraction.py
```
The first command should create an empty file on the first go through.
If the program stops halfway for whatever reason then you can re update list and start again.




## Step 6 : run test code

the actionformer code has no changes for validation

repeat step 3 and then move the features into the correct folder

run it as normal and get the results using 

python ./eval.py <config file> <ckpt folder>

example:
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce


Note:

some features can be downloaded here
https://drive.google.com/file/d/1Wfc38Yau2RuhNls8yXJOBRqEfOTjyZlM/view?usp=sharing
