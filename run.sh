# train VGG16Trans model on FSC
python train.py --resume "./checkpoint/0515_fsc-baseline/100_ckpt.tar" --tag fsc-baseline --device 0 --scheduler step --step 400 --dcsize 8 --batch-size 8 --lr 4e-5 --val-start 100 --val-epoch 10 
