#!/bin/bash -l

#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J jsc
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32

conda activate graph
device=$1

output_folder=results/javasmall_context
bash javasmall_fine_tuned.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh gcn "pretrained_models/context/gcn/model_0" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh gin "pretrained_models/context/gin/model_0" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh graphsage "pretrained_models/context/graphsage/model_0" ${output_folder}/ attention $device

output_folder=results/javasmall_node
bash javasmall_fine_tuned.sh gat "pretrained_models/node/gat/model_0" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh gcn "pretrained_models/node/gcn/model_0" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh gin "pretrained_models/node/gin/model_0" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh graphsage "pretrained_models/node/graphsage/model_0" ${output_folder}/ attention $device

output_folder=results/javasmall_vgae
bash javasmall_fine_tuned.sh gat "pretrained_models/vgae/gat/model_1" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh gcn "pretrained_models/vgae/gcn/model_1" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh gin "pretrained_models/vgae/gin/model_1" ${output_folder}/ attention $device
bash javasmall_fine_tuned.sh graphsage "pretrained_models/vgae/graphsage/model_1" ${output_folder}/ attention $device
