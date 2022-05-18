
## Step 1 : extract features for training set

First, a baseline set of features must be extracted using i3d backbone.

* go to extract.py
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

in stamps.py
* change line 14 to an output folder
* on line 17 have a filename that contains the filenames to look at

run the file stamps.py
```shell
python stamps.py
```

this will extract and save all of the time stamps for the validation videos in a stamps folder






## Step 3 : combine extracted features with optical flow features

run

```shell
ls " folder where validation FEATURES are saved"     > transfer.txt
```

* change lines 8-11 in the setfeatures.py

use setfeatures.py to save the newly extracted features over the old files, where the optical flow data is saved





## Step 4 : run training

set the parameters in the config yaml and then run training

```shell
python ./train.py ./configs/thumos_i3d.yaml --output "name of the checkpoint file you want"
```



## Step 5 : extract test features

The network from step 4 will be saved somewhere as dictated by the yaml
use this network to re extract new features on the test videos

```shell
python testextraction.py
```

*edit lines 12-16 of testextraction.py and then
*do  these commands
```shell
ls "folder of test features" > list.txt
ls "folder of test videos" > todo.txt
python testextraction.py
```





## Step 6 : run test code

the actionformer code has no changes for validation

move the test features into the correct folder

run it as normal and get the results



