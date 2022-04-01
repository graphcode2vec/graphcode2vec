#!/bin/bash
gnn_type=$1
pretrainpath=$2
output_prefix=$3
gp=$4
device=$5
num_class=$6
cd ../../

output=${output_prefix}/${gnn_type}
sw=lstm
jk=sum
lstm_emb_dim=150
#mkdir -p $output
python patchoverfitting_classification.py --batch_size 128 --num_workers 5  --epochs 50 --num_layer 5 \
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
--num_class ${num_class} \
--lr 0.001 \
--dropratio 0.1 \
--warmup_schedule yes \
--mutant_type no \
--lazy yes \
--grid_search no


