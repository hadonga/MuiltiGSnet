#!/bin/bash
echo "Training start!"
python ./train.py -m Our_trans_DSUNet
#python ./train.py -m Our_AUNet
#./train.py -m Our_UNet
echo "Our_trans_DSUNet,Our_AUNet,Our_UNet training completed."