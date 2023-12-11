# FLCMA
## Dependencies
Our model was developed and evaluated using the following package dependencies:
* PyTorch 1.8.1
* OpenCV 4.5.3
* Transformers 4.6.1
* pandas
* ujson
* timm
* ftfy 

## Dataset
Please refer to download data set：https://github.com/ArrowLuo/CLIP4Clip


## Train
### MSRVTT
python train.py  --exp_name={exp_name}  --videos_dir={videos_dir}  --batch_size=32  --noclip_lr=3e-5/1e-5  --transformer_dropout=0.3/0.4   --dataset_name=MSRVTT  --msrvtt_train_file=9k/7k
### MSVD
python train.py  --exp_name={exp_name}  --videos_dir={videos_dir}  --batch_size=32  --noclip_lr=1e-5   --transformer_dropout=0.4     --dataset_name=MSVD

## Test
### MSRVTT
python test.py  --exp_name={exp_name}  --videos_dir={videos_dir}  --batch_size=32  --load_epoch=-1  --dataset_name=MSRVTT  --msrvtt_train_file=9k/7k
### MSVD
python test.py  --exp_name={exp_name}  --videos_dir={videos_dir}  --batch_size=32  --load_epoch=-1  --dataset_name=MSVD
