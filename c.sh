
#cp /scratch/cruggles/features/purethumos/i3d_features/video_valid* ./data/thumos/i3d_features

#python ./train.py ./configs/thumos_i3d.yaml --output "alternating"
#python ./train.py ./configs/thumos_i3d.yaml --output "final"
#python ./train.py ./configs/thumos_i3d.yaml --output "final2" --resume "/home/cruggles/actionformer_release-main/ckpt/thumos_i3d_final/epoch_030.pth.tar" 
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_final/
