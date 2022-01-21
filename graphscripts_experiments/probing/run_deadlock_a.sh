# ### VGE
bash deadlock.sh gat pretrained_models/vgae/gat/model_1.pth $1/deadlock/vge/gat 0 deadlock
bash deadlock.sh gin pretrained_models/vgae/gin/model_1.pth $1/deadlock/vge/gin 0 deadlock
bash deadlock.sh gcn pretrained_models/vgae/gcn/model_1.pth $1/deadlock/vge/gcn 0 deadlock
bash deadlock.sh graphsage pretrained_models/vgae/graphsage/model_1.pth $1/deadlock/vge/graphsage 0 deadlock


### Context
bash deadlock.sh gat pretrained_models/context/gat/model_0.pth $1/deadlock/context/gat 0 deadlock
bash deadlock.sh gin pretrained_models/context//gin/model_0.pth $1/deadlock/context/gin 0 deadlock
bash deadlock.sh gcn pretrained_models/context//gcn/model_0.pth $1/deadlock/context/gcn 0 deadlock
bash deadlock.sh graphsage pretrained_models/context/graphsage/model_0.pth $1/deadlock/context/graphsage 0 deadlock


### Node
bash deadlock.sh gat pretrained_models/node/gat/model_0.pth $1/deadlock/node/gat 0 deadlock
bash deadlock.sh gin pretrained_models/node/gin/model_0.pth $1/deadlock/node/gin 0 deadlock
bash deadlock.sh gcn pretrained_models/node/gcn/model_0.pth $1/deadlock/node/gcn 0 deadlock
bash deadlock.sh graphsage pretrained_models/node/graphsage/model_0.pth $1/deadlock/node/graphsage 0 deadlock

