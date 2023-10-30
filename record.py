quit()
features from from thumos, test features are reextracted every time
epochs: 80
extraction code : extract.py + extract large
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
w/ 793
dropout on 

|tIoU = 0.30: mAP = 74.19 (%)
|tIoU = 0.40: mAP = 69.19 (%)
|tIoU = 0.50: mAP = 61.24 (%)
|tIoU = 0.60: mAP = 51.54 (%)
|tIoU = 0.70: mAP = 39.66 (%)
Avearge mAP: 59.17 (%)



features from from thumos, test features are reextracted every time
epochs: 40
extraction code : extract.py + extract large
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
no 793
dropout on 



|tIoU = 0.30: mAP = 74.26 (%)
|tIoU = 0.40: mAP = 70.90 (%)
|tIoU = 0.50: mAP = 63.39 (%)
|tIoU = 0.60: mAP = 53.78 (%)
|tIoU = 0.70: mAP = 41.00 (%)
Avearge mAP: 60.66 (%)
All done! Total time: 45.67 sec



features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : random chunks
dropout on 



|tIoU = 0.30: mAP = 74.60 (%)
|tIoU = 0.40: mAP = 71.34 (%)
|tIoU = 0.50: mAP = 64.17 (%)
|tIoU = 0.60: mAP = 54.21 (%)
|tIoU = 0.70: mAP = 41.53 (%)
Avearge mAP: 61.17 (%)
All done! Total time: 45.08 sec

features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : random chunks
turned off dropout, 40 clips

|tIoU = 0.30: mAP = 74.62 (%)
|tIoU = 0.40: mAP = 70.05 (%)
|tIoU = 0.50: mAP = 63.51 (%)
|tIoU = 0.60: mAP = 53.29 (%)
|tIoU = 0.70: mAP = 40.81 (%)
Avearge mAP: 60.46 (%)


features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : stratified random
turned off dropout, 40 clips
4 sized clips 


|tIoU = 0.30: mAP = 74.50 (%)
|tIoU = 0.40: mAP = 69.91 (%)
|tIoU = 0.50: mAP = 62.83 (%)
|tIoU = 0.60: mAP = 53.90 (%)
|tIoU = 0.70: mAP = 40.92 (%)
Avearge mAP: 60.41 (%)

features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : stratified random
turned off dropout, 40 clips
1 contiguous

|tIoU = 0.30: mAP = 73.67 (%)
|tIoU = 0.40: mAP = 69.50 (%)
|tIoU = 0.50: mAP = 62.87 (%)
|tIoU = 0.60: mAP = 52.47 (%)
|tIoU = 0.70: mAP = 39.76 (%)
Avearge mAP: 59.65 (%)

features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : stratified random
turned off dropout, 40 clips
4 contiguous, 40 clips
resuming training

|tIoU = 0.30: mAP = 74.22 (%)
|tIoU = 0.40: mAP = 70.06 (%)
|tIoU = 0.50: mAP = 61.99 (%)
|tIoU = 0.60: mAP = 51.15 (%)
|tIoU = 0.70: mAP = 38.58 (%)
Avearge mAP: 59.20 (%)



features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : stratified random
turned off dropout, 40 clips
4 contiguous, 40 clips, re train af from scratch after feature extraction
|tIoU = 0.30: mAP = 80.41 (%) Recall@1x = 83.54 (%) Recall@5x = 96.00 (%)
|tIoU = 0.40: mAP = 75.25 (%) Recall@1x = 78.76 (%) Recall@5x = 94.40 (%)
|tIoU = 0.50: mAP = 67.42 (%) Recall@1x = 72.32 (%) Recall@5x = 90.44 (%)
|tIoU = 0.60: mAP = 55.86 (%) Recall@1x = 62.88 (%) Recall@5x = 82.50 (%)
|tIoU = 0.70: mAP = 41.19 (%) Recall@1x = 51.00 (%) Recall@5x = 68.46 (%)
Average mAP: 64.03 (%)

===================================================================================

features from from thumos, test features are reextracted every time
epochs: 40
extraction code : alex
spatial augmentation : none
learning rate 1e-4
extractor learning rate 1e-5
clips = 40
793 included
sample method : stratified random
turned off dropout, 40 clips
1 contiguous, 40 clips, re train af from scratch after feature extraction

|tIoU = 0.30: mAP = 80.06 (%) Recall@1x = 82.33 (%) Recall@5x = 95.98 (%)
|tIoU = 0.40: mAP = 74.99 (%) Recall@1x = 78.15 (%) Recall@5x = 93.88 (%)
|tIoU = 0.50: mAP = 67.43 (%) Recall@1x = 72.11 (%) Recall@5x = 91.01 (%)
|tIoU = 0.60: mAP = 55.89 (%) Recall@1x = 62.38 (%) Recall@5x = 82.87 (%)
|tIoU = 0.70: mAP = 40.95 (%) Recall@1x = 50.55 (%) Recall@5x = 69.12 (%)
Average mAP: 63.86 (%)
All done! Total time: 32.82 sec
===================================================================================

same 
but with action centric sampling 

|tIoU = 0.30: mAP = 80.36 (%) Recall@1x = 83.18 (%) Recall@5x = 96.27 (%)
|tIoU = 0.40: mAP = 74.78 (%) Recall@1x = 78.30 (%) Recall@5x = 94.36 (%)
|tIoU = 0.50: mAP = 67.08 (%) Recall@1x = 71.96 (%) Recall@5x = 90.63 (%)
|tIoU = 0.60: mAP = 55.67 (%) Recall@1x = 62.56 (%) Recall@5x = 82.30 (%)
|tIoU = 0.70: mAP = 41.64 (%) Recall@1x = 50.76 (%) Recall@5x = 69.21 (%)
Average mAP: 63.91 (%)
All done! Total time: 34.31 sec
===================================================================================


same w/  non action centric sampling
and with model ema

|tIoU = 0.30: mAP = 79.77 (%) Recall@1x = 83.23 (%) Recall@5x = 95.75 (%)
|tIoU = 0.40: mAP = 74.41 (%) Recall@1x = 78.47 (%) Recall@5x = 93.61 (%)
|tIoU = 0.50: mAP = 66.61 (%) Recall@1x = 71.62 (%) Recall@5x = 90.06 (%)
|tIoU = 0.60: mAP = 55.53 (%) Recall@1x = 62.63 (%) Recall@5x = 81.97 (%)
|tIoU = 0.70: mAP = 40.32 (%) Recall@1x = 49.70 (%) Recall@5x = 68.54 (%)
Average mAP: 63.33 (%)
All done! Total time: 34.18 sec


===================================================================================

block sampling wout/ feature ema


|tIoU = 0.30: mAP = 80.58 (%) Recall@1x = 83.65 (%) Recall@5x = 95.98 (%)
|tIoU = 0.40: mAP = 75.44 (%) Recall@1x = 79.11 (%) Recall@5x = 94.27 (%)
|tIoU = 0.50: mAP = 67.49 (%) Recall@1x = 72.54 (%) Recall@5x = 90.55 (%)
|tIoU = 0.60: mAP = 56.05 (%) Recall@1x = 62.95 (%) Recall@5x = 83.12 (%)
|tIoU = 0.70: mAP = 40.89 (%) Recall@1x = 50.76 (%) Recall@5x = 69.30 (%)
Average mAP: 64.09 (%


block sampling w/ feature ema
|tIoU = 0.30: mAP = 80.30 (%) Recall@1x = 82.89 (%) Recall@5x = 96.04 (%)
|tIoU = 0.40: mAP = 74.55 (%) Recall@1x = 78.44 (%) Recall@5x = 94.34 (%)
|tIoU = 0.50: mAP = 67.66 (%) Recall@1x = 72.96 (%) Recall@5x = 90.65 (%)
|tIoU = 0.60: mAP = 56.75 (%) Recall@1x = 63.54 (%) Recall@5x = 83.06 (%)
|tIoU = 0.70: mAP = 41.66 (%) Recall@1x = 51.23 (%) Recall@5x = 69.27 (%)
Average mAP: 64.19 (%)


extractor LR = 1e-4 instead of 1e-5

|tIoU = 0.30: mAP = 79.97 (%) Recall@1x = 83.55 (%) Recall@5x = 96.13 (%)
|tIoU = 0.40: mAP = 75.23 (%) Recall@1x = 79.30 (%) Recall@5x = 94.58 (%)
|tIoU = 0.50: mAP = 67.32 (%) Recall@1x = 72.97 (%) Recall@5x = 90.93 (%)
|tIoU = 0.60: mAP = 55.28 (%) Recall@1x = 62.92 (%) Recall@5x = 82.73 (%)
|tIoU = 0.70: mAP = 41.08 (%) Recall@1x = 51.29 (%) Recall@5x = 69.43 (%)
Average mAP: 63.77 (%)
All done! Total time: 33.33 sec

no LR decay:
"fixedlr"
|tIoU = 0.30: mAP = 80.49 (%) Recall@1x = 82.62 (%) Recall@5x = 96.29 (%)
|tIoU = 0.40: mAP = 75.64 (%) Recall@1x = 78.68 (%) Recall@5x = 94.57 (%)
|tIoU = 0.50: mAP = 68.22 (%) Recall@1x = 72.78 (%) Recall@5x = 90.67 (%)
|tIoU = 0.60: mAP = 56.28 (%) Recall@1x = 62.80 (%) Recall@5x = 82.50 (%)
|tIoU = 0.70: mAP = 41.60 (%) Recall@1x = 51.09 (%) Recall@5x = 69.50 (%)
Average mAP: 64.45 (%)


more frozen layers
[RESULTS] Action detection results on thumos14.

|tIoU = 0.30: mAP = 79.72 (%) Recall@1x = 82.76 (%) Recall@5x = 96.14 (%)
|tIoU = 0.40: mAP = 74.20 (%) Recall@1x = 77.91 (%) Recall@5x = 94.35 (%)
|tIoU = 0.50: mAP = 66.27 (%) Recall@1x = 71.49 (%) Recall@5x = 90.37 (%)
|tIoU = 0.60: mAP = 55.11 (%) Recall@1x = 62.52 (%) Recall@5x = 81.91 (%)
|tIoU = 0.70: mAP = 40.53 (%) Recall@1x = 50.21 (%) Recall@5x = 67.93 (%)
Average mAP: 63.17 (%)




less frozen layers, no LR decay, block sampling w/ feature EMa, EPOCHS 50, retraining
"neckdeepthumos"
|tIoU = 0.30: mAP = 81.27 (%) Recall@1x = 83.21 (%) Recall@5x = 96.24 (%)
|tIoU = 0.40: mAP = 75.52 (%) Recall@1x = 78.70 (%) Recall@5x = 94.24 (%)
|tIoU = 0.50: mAP = 68.07 (%) Recall@1x = 72.28 (%) Recall@5x = 90.71 (%)
|tIoU = 0.60: mAP = 55.78 (%) Recall@1x = 62.79 (%) Recall@5x = 82.80 (%)
|tIoU = 0.70: mAP = 42.21 (%) Recall@1x = 51.65 (%) Recall@5x = 69.73 (%)
Average mAP: 64.57 (%)

re video extraction

|tIoU = 0.30: mAP = 81.25 (%) Recall@1x = 83.42 (%) Recall@5x = 96.36 (%)                                                                                                                                                                                                                                                                                                           
|tIoU = 0.40: mAP = 75.76 (%) Recall@1x = 78.47 (%) Recall@5x = 94.73 (%)                                                                                                                                                                                                                                                                                                           
|tIoU = 0.50: mAP = 67.83 (%) Recall@1x = 72.15 (%) Recall@5x = 90.98 (%)                                                                                                                                                                                                                                                                                                           
|tIoU = 0.60: mAP = 56.48 (%) Recall@1x = 63.40 (%) Recall@5x = 82.69 (%)                                                                                                                                                                                                                                                                                                           
|tIoU = 0.70: mAP = 41.54 (%) Recall@1x = 50.94 (%) Recall@5x = 68.23 (%)                                                                                                                                                                                                                                                                                                           
Average mAP: 64.57 (%)   

alex features?

|tIoU = 0.30: mAP = 80.45 (%) Recall@1x = 82.86 (%) Recall@5x = 96.31 (%)
|tIoU = 0.40: mAP = 75.00 (%) Recall@1x = 78.07 (%) Recall@5x = 95.04 (%)
|tIoU = 0.50: mAP = 68.18 (%) Recall@1x = 72.28 (%) Recall@5x = 91.63 (%)
|tIoU = 0.60: mAP = 56.70 (%) Recall@1x = 63.42 (%) Recall@5x = 83.51 (%)
|tIoU = 0.70: mAP = 42.54 (%) Recall@1x = 52.04 (%) Recall@5x = 70.21 (%)
Average mAP: 64.57 (%)

pure thumos

|tIoU = 0.30: mAP = 81.06 (%) Recall@1x = 83.10 (%) Recall@5x = 96.38 (%)
|tIoU = 0.40: mAP = 76.06 (%) Recall@1x = 78.50 (%) Recall@5x = 94.66 (%)
|tIoU = 0.50: mAP = 68.59 (%) Recall@1x = 72.45 (%) Recall@5x = 91.63 (%)
|tIoU = 0.60: mAP = 57.01 (%) Recall@1x = 63.10 (%) Recall@5x = 82.89 (%)
|tIoU = 0.70: mAP = 43.68 (%) Recall@1x = 52.11 (%) Recall@5x = 70.46 (%)
Average mAP: 65.28 (%)

mythumos
65.12


lots of video updates
|tIoU = 0.30: mAP = 80.66 (%) Recall@1x = 83.18 (%) Recall@5x = 96.03 (%)
|tIoU = 0.40: mAP = 75.40 (%) Recall@1x = 78.98 (%) Recall@5x = 94.22 (%)
|tIoU = 0.50: mAP = 68.35 (%) Recall@1x = 73.30 (%) Recall@5x = 90.60 (%)
|tIoU = 0.60: mAP = 55.37 (%) Recall@1x = 62.37 (%) Recall@5x = 82.82 (%)
|tIoU = 0.70: mAP = 40.89 (%) Recall@1x = 50.56 (%) Recall@5x = 69.29 (%)
Average mAP: 64.13 (%)



no video udpate setting
no retrianing


|tIoU = 0.30: mAP = 74.11 (%)
|tIoU = 0.40: mAP = 68.87 (%)
|tIoU = 0.50: mAP = 62.05 (%)
|tIoU = 0.60: mAP = 51.41 (%)
|tIoU = 0.70: mAP = 39.86 (%)
Avearge mAP: 59.26 (%)
LOL

no video update
with retraining, video re extract setting otherwise
|tIoU = 0.30: mAP = 81.05 (%) Recall@1x = 83.47 (%) Recall@5x = 96.19 (%) 
|tIoU = 0.40: mAP = 75.32 (%) Recall@1x = 78.54 (%) Recall@5x = 94.75 (%) 
|tIoU = 0.50: mAP = 67.83 (%) Recall@1x = 72.44 (%) Recall@5x = 90.95 (%) 
|tIoU = 0.60: mAP = 55.83 (%) Recall@1x = 62.48 (%) Recall@5x = 82.78 (%) 
|tIoU = 0.70: mAP = 41.15 (%) Recall@1x = 50.44 (%) Recall@5x = 68.47 (%) 
Average mAP: 64.24 (%)



alternating scheme

|tIoU = 0.30: mAP = 80.51 (%) Recall@1x = 83.21 (%) Recall@5x = 96.34 (%)
|tIoU = 0.40: mAP = 75.70 (%) Recall@1x = 79.38 (%) Recall@5x = 94.80 (%)
|tIoU = 0.50: mAP = 67.73 (%) Recall@1x = 72.24 (%) Recall@5x = 90.93 (%)
|tIoU = 0.60: mAP = 56.28 (%) Recall@1x = 62.17 (%) Recall@5x = 83.10 (%)
|tIoU = 0.70: mAP = 41.53 (%) Recall@1x = 50.31 (%) Recall@5x = 69.08 (%)
Average mAP: 64.35 (%)
