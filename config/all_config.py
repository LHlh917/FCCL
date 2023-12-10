import os
import argparse
from config.base_config import Config
# from modules.basic_utils import mkdirp, deletedir
import shutil

def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)

def deletedir(p):
    if os.path.exists(p):
        shutil.rmtree(p)


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters
        parser.add_argument('--dataset_name', type=str, default='MSVD', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='/opt/data/private/LH/MSVD1', help="Location of videos")
        # parser.add_argument('--videos_dir', type=str, default='/opt/data/private/LH/LH/xpool1/MSRVTT/videos/all', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")    # 没10次打印一次
        parser.add_argument('--evals_per_epoch', type=int, default=10, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, default=-1,help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        # parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")
        parser.add_argument('--metric', type=str, default='v2t', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=False)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=3e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.4, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        ### 后面添加的
        # parser.add_argument('--distributed', action='store_true', help="Enable distributed training")
        # # 添加参数 config.rank
        # parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

        # # 添加参数 config.world_size
        # parser.add_argument('--world_size', default=2,type=int, help="Total number of processes in distributed training")
        # parser.add_argument('--nproc_per_node', default=2,type=int, help="")
        # parser.add_argument('--use_env', action='store_true', help="")
        # 添加参数 config.gpu
        # parser.add_argument('--gpu', default=[0,1],type=int, help="GPU number for the current process")

        # 添加参数 config.dist_backend
        # parser.add_argument('--dist_backend',default="nccl" ,type=str, help="Communication backend (e.g., nccl)")
        # parser.add_argument('--device', default='cuda',type=str, help="Device for training (e.g., 'cuda', 'cpu')")
        # parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        # parser.add_argument('--nproc_per_node', default=2, help='url used to set up distributed training')

        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
