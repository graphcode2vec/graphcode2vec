#gnn_type=$1
#modelpath=$2
#output_prefix=$3
#device=$4

mkdir javasmallperformance
mkdir javasmallperformance/context
for i in $(seq 3)
do
bash javasmall_eval_myappraoch.sh gat  javasmall_p0_context_new/gat_${i}/saved_model.pt javasmallperformance/context_${i}/ 0 2>&1 | tee javasmallperformance/logp_contextgat_${i}.txt 
bash javasmall_eval_myappraoch.sh gin  javasmall_p0_context_new/gin_${i}/saved_model.pt javasmallperformance/context_${i}/ 0 2>&1 | tee  javasmallperformance/logp_contextgin_${i}.txt
bash javasmall_eval_myappraoch.sh gcn  javasmall_p0_context_new/gcn_${i}/saved_model.pt javasmallperformance/context_${i}/ 0 2>&1 | tee  javasmallperformance/logp_contextgcn_${i}.txt
bash javasmall_eval_myappraoch.sh graphsage  javasmall_p0_context_new/graphsage_${i}/saved_model.pt javasmallperformance/context_${i}/ 0 2>&1 | tee  javasmallperformance/logp_contextgraphsage_${i}.txt
done

mkdir javasmallperformance/node
for i in $(seq 3)
do
bash javasmall_eval_myappraoch.sh gat  javasmall_p0_nodepretrain_new/gat_${i}/saved_model.pt javasmallperformance/node_${i}/ 0 2>&1 | tee javasmallperformance/logp_nodegat_${i}.txt 
bash javasmall_eval_myappraoch.sh gin  javasmall_p0_nodepretrain_new/gin_${i}/saved_model.pt javasmallperformance/node_${i}/ 0 2>&1 | tee  javasmallperformance/logp_nodegin_${i}.txt
bash javasmall_eval_myappraoch.sh gcn  javasmall_p0_nodepretrain_new/gcn_${i}/saved_model.pt javasmallperformance/node_${i}/ 0 2>&1 | tee  javasmallperformance/logp_nodegcn_${i}.txt
bash javasmall_eval_myappraoch.sh graphsage  javasmall_p0_nodepretrain_new/graphsage_${i}/saved_model.pt javasmallperformance/node_${i}/ 0 2>&1 | tee  javasmallperformance/logp_nodegraphsage_${i}.txt
done

mkdir javasmallperformance/vge
for i in $(seq 3)
do
bash javasmall_eval_myappraoch.sh gat  javasmall_p0_pretrain_new/gat_${i}/saved_model.pt javasmallperformance/vge_${i}/ 0 2>&1 | tee javasmallperformance/logp_vgegat_${i}.txt 
bash javasmall_eval_myappraoch.sh gin  javasmall_p0_pretrain_new/gin_${i}/saved_model.pt javasmallperformance/vge_${i}/ 0 2>&1 | tee  javasmallperformance/logp_vgegin_${i}.txt
bash javasmall_eval_myappraoch.sh gcn  javasmall_p0_pretrain_new/gcn_${i}/saved_model.pt javasmallperformance/vge_${i}/ 0 2>&1 | tee  javasmallperformance/logp_vgegcn_${i}.txt
bash javasmall_eval_myappraoch.sh graphsage  javasmall_p0_pretrain_new/graphsage_${i}/saved_model.pt javasmallperformance/vge_${i}/ 0 2>&1 | tee  javasmallperformance/logp_vgegraphsage_${i}.txt
done