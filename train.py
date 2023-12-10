import os
import sys
import time
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from model.metrics import t2v_metrics, v2t_metrics
from model.loss import LossFactory
from trainer.trainer import Trainer
from model.optimization import AdamW, get_cosine_schedule_with_warmup

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import BatchSampler


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None
    # print(os.environ.get('RANK', 0))
    # rank = int(os.environ.get('RANK', 0))
    # world_size = torch.cuda.device_count()
    # torch.distributed.init_process_group(backend="nccl",world_size=world_size,rank=rank)
    # # dist.barrier()
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
        tokenizer = CLIPTokenizer.from_pretrained("openaiclip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from model.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    
    
    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    # model = model.to(device)
    # model = nn.DataParallel(model, device_ids=[0,1])

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # GPU 数目大于 1 才有必要分布式训练
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                     device_ids=[local_rank],
    #                                                     output_device=0,find_unused_parameters=True)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    print(config.metric)
    print(config.dataset_name)
    print(config.noclip_lr)
    print(config.transformer_dropout)
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
      
    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]
    
    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)

    trainer.train()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execute_time = end_time - start_time
    print(f"Execution time: {execute_time} seconds")
