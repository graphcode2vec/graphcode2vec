#!/bin/bash

device=$1
i=node_class_pretraining
for gnn_type in gat gin gcn graphsage
do
#gnn_type=gat
gp=attention
output=${i}_${gp}/${gnn_type}
sw=lstm
jk=sum
lstm_emb_dim=150
mkdir -p $output
python node_vgage_pretrain.py --batch_size 500 --num_workers 5  --epochs 10 --num_layer 5 \
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
--task node_class \
--device $device
done
#--check_all \

i=node_class_pretraining
for gnn_type in gat gin gcn graphsage
do
#gnn_type=gat
sw=lstm
gp=mean
output=${i}_${gp}/${gnn_type}
jk=sum
lstm_emb_dim=150
mkdir -p $output
python node_vgage_pretrain.py --batch_size 500 --num_workers 5  --epochs 10 --num_layer 5 \
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
--task node_class \
--device $device
done




i=node_class_pretraining_all
for gnn_type in gat gin gcn graphsage
do
#gnn_type=gat
gp=attention
output=${i}_${gp}/${gnn_type}
sw=lstm
jk=sum
lstm_emb_dim=150
mkdir -p $output
python node_vgage_pretrain.py --batch_size 500 --num_workers 5  --epochs 10 --num_layer 5 \
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
--task node_class \
--device $device \
--check_all 
done
#--check_all \

i=node_class_pretraining_all
for gnn_type in gat gin gcn graphsage
do
#gnn_type=gat
sw=lstm
gp=mean
output=${i}_${gp}/${gnn_type}
jk=sum
lstm_emb_dim=150
mkdir -p $output
python node_vgage_pretrain.py --batch_size 500 --num_workers 5  --epochs 10 --num_layer 5 \
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
--task node_class \
--device $device \
--check_all 
done