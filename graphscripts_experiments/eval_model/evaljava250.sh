#gnn_type=$1
#modelpath=$2
#output_prefix=$3
#device=$4

mkdir java250performance

for i in $(seq 3)
do
bash java250_eval_myappraoch.sh gat  java250_p0_contextpretrain_new/gat_${i}/saved_model.pt java250performance/context_${i}/ 0 2>&1 | tee java250performance/logp_contextgat_${i}.txt 
bash java250_eval_myappraoch.sh gin  java250_p0_contextpretrain_new/gin_${i}/saved_model.pt java250performance/context_${i}/ 0 2>&1 | tee  java250performance/logp_contextgin_${i}.txt
bash java250_eval_myappraoch.sh gcn  java250_p0_contextpretrain_new/gcn_${i}/saved_model.pt java250performance/context_${i}/ 0 2>&1 | tee  java250performance/logp_contextgcn_${i}.txt
bash java250_eval_myappraoch.sh graphsage  java250_p0_contextpretrain_new/graphsage_${i}/saved_model.pt java250performance/context_${i}/ 0 2>&1 | tee  java250performance/logp_contextgraphsage_${i}.txt
done


for i in $(seq 3)
do
bash java250_eval_myappraoch.sh gat  java250_p0_nodepretrain_new/gat_${i}/saved_model.pt java250performance/node_${i}/ 0 2>&1 | tee java250performance/logp_nodegat_${i}.txt 
bash java250_eval_myappraoch.sh gin  java250_p0_nodepretrain_new/gin_${i}/saved_model.pt java250performance/node_${i}/ 0 2>&1 | tee  java250performance/logp_nodegin_${i}.txt
bash java250_eval_myappraoch.sh gcn  java250_p0_nodepretrain_new/gcn_${i}/saved_model.pt java250performance/node_${i}/ 0 2>&1 | tee  java250performance/logp_nodegcn_${i}.txt
bash java250_eval_myappraoch.sh graphsage  java250_p0_nodepretrain_new/graphsage_${i}/saved_model.pt java250performance/node_${i}/ 0 2>&1 | tee  java250performance/logp_nodegraphsage_${i}.txt
done


for i in $(seq 3)
do
bash java250_eval_myappraoch.sh gat  java250_p0_pretrain_new/gat_${i}/saved_model.pt java250performance/vge_${i}/ 0 2>&1 | tee java250performance/logp_vgegat_${i}.txt 
bash java250_eval_myappraoch.sh gin  java250_p0_pretrain_new/gin_${i}/saved_model.pt java250performance/vge_${i}/ 0 2>&1 | tee  java250performance/logp_vgegin_${i}.txt
bash java250_eval_myappraoch.sh gcn  java250_p0_pretrain_new/gcn_${i}/saved_model.pt java250performance/vge_${i}/ 0 2>&1 | tee  java250performance/logp_vgegcn_${i}.txt
bash java250_eval_myappraoch.sh graphsage  java250_p0_pretrain_new/graphsage_${i}/saved_model.pt java250performance/vge_${i}/ 0 2>&1 | tee  java250performance/logp_vgegraphsage_${i}.txt
done