from config.base_config import Config
import sys
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":

            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=config.world_size)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
                # return DataLoader(dataset, pin_memory=True,batch_size=config.batch_size,
                #             num_workers=config.num_workers,sampler=train_sampler)
                                                  # 直接加载到显存中，达到加速效果)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False,num_replicas=config.world_size)
                # return DataLoader(dataset, batch_size=config.batch_size,pin_memory=True,
                #             num_workers=config.num_workers,sampler=test_sampler)
                return DataLoader(dataset, batch_size=config.batch_size,shuffle=False,
                            num_workers=config.num_workers)

        else:
            raise NotImplementedError
