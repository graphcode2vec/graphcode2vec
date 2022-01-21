#!/bin/bash
gnn_type=$1
modelpath=$2
output_prefix=$3
device=$4
output=${output_prefix}/${gnn_type}
sw=lstm
gp=attention
jk=sum
lstm_emb_dim=150
mkdir -p $output
python java250_classification_eval.py --batch_size 500 --num_workers 3  --epochs 20 --num_layer 5 \
--subword_embedding  $sw \
--lstm_emb_dim $lstm_emb_dim \
--graph_pooling $gp \
--JK $jk \
--saved_model_path $modelpath \
--log_file ${output}/log.txt \
--gnn_type $gnn_type \
--sub_token_path ./tokens/jars \
--emb_file emb_100.txt \
--dataset DV_PDG \
--savedperformacenfile ${output}/evalperformance.json \
--device $device