#!/bin/bash -l

#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J j2_con_graph
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32

conda activate graph
rep=graph
device=$1

output_folder=results/ablation_java250_context

bash java250_ablation_finetuned.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh gcn "pretrained_models/context/gcn/model_0" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh gin "pretrained_models/context/gin/model_0" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh graphsage "pretrained_models/context/graphsage/model_0" ${output_folder}/${rep} attention ${rep} ${device}


output_folder=results/ablation_java250_node

bash java250_ablation_finetuned.sh gat "pretrained_models/node/gat/model_0" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh graphsage "pretrained_models/node/graphsage/model_0" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh gin "pretrained_models/node/gin/model_0" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh gcn "pretrained_models/node/gcn/model_0" ${output_folder}/${rep} attention ${rep} ${device}


output_folder=results/ablation_java250_vgae

bash java250_ablation_finetuned.sh gat "pretrained_models/vgae/gat/model_1" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh gcn "pretrained_models/vgae/gcn/model_1" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh gin "pretrained_models/vgae/gin/model_1" ${output_folder}/${rep} attention ${rep} ${device}
bash java250_ablation_finetuned.sh graphsage "pretrained_models/vgae/graphsage/model_1" ${output_folder}/${rep} attention ${rep} ${device}




