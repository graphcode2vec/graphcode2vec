# ### VGE
arg=$(realpath $1)
num_class=$2
# bash mutants.sh gat pretrained_models/vgae/gat/model_1.pth $arg/mutants/vge/gat 0  2>&1 > $arg/mutants/vge/gat/mutaints_probing.log | tee $arg/mutants/vge/gat/mutaints_probing.log
# bash mutants.sh gin pretrained_models/vgae/gin/model_1.pth $arg/mutants/vge/gin 0  2>&1 > $arg/mutants/vge/gin/mutaints_probing.log  | tee $arg/mutants/vge/gin/mutaints_probing.log
# bash mutants.sh gcn pretrained_models/vgae/gcn/model_1.pth $arg/mutants/vge/gcn 0  2>&1 > $arg/mutants/vge/gcn/mutaints_probing.log  | tee $arg/mutants/vge/gcn/mutaints_probing.log
# bash mutants.sh graphsage pretrained_models/vgae/graphsage/model_1.pth $arg/mutants/vge/graphsage 0  2>&1 > $arg/mutants/vge/graphsage/mutaints_probing.log  | tee $arg/mutants/vge/graphsage/mutaints_probing.log

### Context
mkdir -p  $arg/mutants/context/gat
mkdir -p  $arg/mutants/context/gin
mkdir -p  $arg/mutants/context/gcn
mkdir -p  $arg/mutants/context/graphsage
bash mutants.sh gat pretrained_models/context/gat/model_0.pth $arg/mutants/context/gat 0  $num_class 2>&1 > $arg/mutants/context/gat/mutaints_probing.log  | tee $arg/mutants/context/gat/mutaints_probing.log
bash mutants.sh gin pretrained_models/context/gin/model_0.pth $arg/mutants/context/gin 0 $num_class 2>&1 > $arg/mutants/context/gin/mutaints_probing.log  | tee $arg/mutants/context/gin/mutaints_probing.log
bash mutants.sh gcn pretrained_models/context/gcn/model_0.pth $arg/mutants/context/gcn 0 $num_class 2>&1 > $arg/mutants/context/gcn/mutaints_probing.log  | tee $arg/mutants/context/gcn/mutaints_probing.log
bash mutants.sh graphsage pretrained_models/context/graphsage/model_0.pth $arg/mutants/context/graphsage 0 $num_class 2>&1 > $arg/mutants/context/graphsage/mutaints_probing.log  | tee $arg/mutants/context/graphsage/mutaints_probing.log


### Node
mkdir -p  $arg/mutants/node/gat
mkdir -p  $arg/mutants/node/gin
mkdir -p  $arg/mutants/node/gcn
mkdir -p  $arg/mutants/node/graphsage
bash mutants.sh gat pretrained_models/node/gat/model_0.pth $arg/mutants/node/gat 0 $num_class 2>&1 > $arg/mutants/node/gat/mutaints_probing.log  | tee $arg/mutants/node/gat/mutaints_probing.log
bash mutants.sh gin pretrained_models/node/gin/model_0.pth $arg/mutants/node/gin 0 $num_class 2>&1 > $arg/mutants/node/gin/mutaints_probing.log  | tee $arg/mutants/node/gin/mutaints_probing.log
bash mutants.sh gcn pretrained_models/node/gcn/model_0.pth $arg/mutants/node/gcn 0 $num_class 2>&1 > $arg/mutants/node/gcn/mutaints_probing.log  | tee $arg/mutants/node/gcn/mutaints_probing.log
bash mutants.sh graphsage pretrained_models/node/graphsage/model_0.pth $arg/mutants/node/graphsage 0 $num_class 2>&1 > $arg/mutants/node/graphsage/mutaints_probing.log  | tee $arg/mutants/node/graphsage/mutaints_probing.log

### Vgae
mkdir -p  $arg/mutants/vgae/gat
mkdir -p  $arg/mutants/vgae/gin
mkdir -p  $arg/mutants/vgae/gcn
mkdir -p  $arg/mutants/vgae/graphsage
bash mutants.sh gat pretrained_models/vgae/gat/model_1.pth $arg/mutants/vgae/gat 0 $num_class 2>&1 > $arg/mutants/vgae/gat/mutaints_probing.log  | tee $arg/mutants/vgae/gat/mutaints_probing.log
bash mutants.sh gin pretrained_models/vgae/gin/model_1.pth $arg/mutants/vgae/gin 0 $num_class 2>&1 > $arg/mutants/vgae/gin/mutaints_probing.log  | tee $arg/mutants/vgae/gin/mutaints_probing.log
bash mutants.sh gcn pretrained_models/vgae/gcn/model_1.pth $arg/mutants/vgae/gcn 0 $num_class 2>&1 > $arg/mutants/vgae/gcn/mutaints_probing.log  | tee $arg/mutants/vgae/gcn/mutaints_probing.log
bash mutants.sh graphsage pretrained_models/vgae/graphsage/model_1.pth $arg/mutants/vgae/graphsage 0 $num_class 2>&1 > $arg/mutants/vgae/graphsage/mutaints_probing.log  | tee $arg/mutants/vgae/graphsage/mutaints_probing.log
