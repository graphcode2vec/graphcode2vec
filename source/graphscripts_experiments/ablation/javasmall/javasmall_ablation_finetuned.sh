#!/bin/bash
gnn_type=$1
pretrainpath=$2
output_prefix=$3
gp=$4
rep=$5
device=$6
cd ../../../
for j in $(seq 3)
do
output=${output_prefix}/${gnn_type}_${j}
sw=lstm
#gp=attention

jk=sum
lstm_emb_dim=150
mkdir -p $output
python method_name_prediction.py --batch_size 500 --num_workers 3  --epochs 20 --num_layer 5 \
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
--repWay $rep \
--device $device
done 

