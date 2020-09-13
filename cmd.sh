#!/usr/bin/env bash
GPU=${1:-0}
mode=${2:-train}
exp_name=${3:-my_exp}

echo ${mode}
if [ ${mode} == 'train' ]
then
    data_path=data/cp_change
elif [ ${mode} == 'test' ]
then
    data_path=data
fi

# Parameter for RNN TEL, GNN SEL
CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
    --spatial-encoding-layer gnn --temporal-encoding-layer rnn \
    --exp_dir exp/${exp_name}_rnn_gnn

# Parameter for TRANS TEL, GNN SEL
#CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer gnn --temporal-encoding-layer transformer \
#    --encoder-hidden 64 --decoder mlp --batch-size 32  \
#    --exp_dir exp/${exp_name}_trans_gnn


# Parameter for RNN TEL, TRANS SEL
#CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer trans --temporal-encoding-layer rnn \
#    --encoder-hidden 64 --decoder mlp --batch-size 32  \
#    --exp_dir exp/${exp_name}_rnn_trans