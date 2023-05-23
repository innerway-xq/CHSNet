# train VGG16Trans model on FSC
python train.py --no-wandb --tag counttr --device 1 --scheduler step --step 400 --dcsize 8 --batch-size 8 --lr 4e-5 --val-start 0 --val-epoch 5 --max-epoch 50

