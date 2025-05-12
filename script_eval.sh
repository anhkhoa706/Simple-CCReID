# Change the Model.Resume to the path of the model you want to test
# Change the --dataset to the dataset you want to test
python main_single_gpu.py --dataset prcc --cfg configs/res50_cels_cal.yaml --eval --gpu 0
