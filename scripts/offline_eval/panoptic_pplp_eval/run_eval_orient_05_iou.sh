#!/bin/bash

# args = [make_script, script_folder, str(score_threshold), str(global_step), str(checkpoint_name), str(results_dir)])
# make_script =  /home/boom/LidarPoseBEV/src/PPLP/pplp/data/outputs/pplp_pedestrian_panoptic/predictions/panoptic_pplp_eval/run_eval_orient_05_iou.sh
# script_folder =  /home/boom/LidarPoseBEV/src/PPLP/pplp/data/outputs/pplp_pedestrian_panoptic/predictions/panoptic_pplp_eval/
# score_threshold =  0.10000000149011612
# global_step =  90000
# checkpoint_name =  pplp_pedestrian_panoptic
# results_dir =  /home/boom/LidarPoseBEV/src/PPLP/scripts/offline_eval/results_orient_05_iou/
set -e

cd $1
echo "$3" | tee -a ./$4_results_orient_05_iou_$2.txt

./evaluate_orient_offline_05_iou /home/boom/Panoptic/projects/bodypose2dsim/160422_ultimatum1/training/label_2 $2/$3 | tee -a ./$4_results_orient_05_iou_$2.txt

cp $4_results_orient_05_iou_$2.txt $5
