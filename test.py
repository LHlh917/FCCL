import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer
import torch.distributed as dist

def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    # if 'RANK' in os.environ:
    #     rank = int(os.environ['RANK'])
    # torch.distributed.init_process_group(backend="nccl")
    # dist.barrier()
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)   
    # device = torch.device("cuda", local_rank)



    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("/opt/data/private/LH/LH/xpool1/openaiclip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    # model = model.to(device)
    # model = nn.DataParallel(model, device_ids=[0,1])

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # GPU 数目大于 1 才有必要分布式训练
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                     device_ids=[local_rank],
    #                                                     output_device=local_rank,find_unused_parameters=True)
    print(config.metric)
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        print(config.load_epoch)
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("/opt/data/private/LH/LH-v2t/xpool1/outputs/{MSVD_v2t}/model_best.pth")    
    trainer.validate()


if __name__ == '__main__':
    main()

