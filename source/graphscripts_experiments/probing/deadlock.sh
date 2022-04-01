#!/bin/bash
cd ../../
gnn_type=$1
pretrainpath=$2
output_prefix=$3
gp=attention
device=$4
data=$5
output=${output_prefix}
sw=lstm
#gp=attention
jk=sum
lstm_emb_dim=150
mkdir -p $output
python probing_analysis.py --batch_size 200 --num_workers 5  --epochs 100 --num_layer 5 \
--subword_embedding  $sw \
--lstm_emb_dim $lstm_emb_dim \
--graph_pooling $gp \
--JK $jk \
--saved_model_path ${output} \
--log_file ${output}/log.txt \
--gnn_type $gnn_type \
--sub_token_path ./tokens/jars \
--emb_file emb_100.txt \
--dataset DV_PDG \
--input_model_file ${pretrainpath} \
--device ${device} \
--data $data \
--epochs 1 \
--k_fold 3 \
--dataset_path dataset/deadlock/

 

