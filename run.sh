# train VGG16Trans model on FSC
python train.py --tag reslink --device 2 --scheduler step --step 400 --dcsize 8 --batch-size 8 --lr 4e-5 --val-start 100 --val-epoch 10 --max-epoch 200
