# train VGG16Trans model on FSC
python train.py --tag cross_vgg --device 2 --scheduler step --step 400 --dcsize 8 --batch-size 8 --lr 2e-5 --val-start 50 --val-epoch 10 --max-epoch 250

