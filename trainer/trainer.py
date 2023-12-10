from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from model.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id,generate_embeds_per_video_id1,generate_embeds_per_video_id2,generate_embeds_per_video_id3
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 



        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        # eval_steps = np.linspace(0, num_steps-1, 2600, dtype=int)[1:]
        trainloader = self.train_data_loader
        # trainloader.sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(trainloader):
        # for  data,_ in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
                # data['word'] = self.tokenizer(data['word'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
                # data['text'] = data['text'].cuda(self.local_rank)
            # else:
            #     data['text'][0] = {key: val.to(self.device) for key, val in data['text'][0]}
            #     data['text'][1] = {key: val.to(self.device) for key, val in data['text'][1]}
                # data['text'][0] = data['text'][0].to(self.device)
                # data['text'][1] = data['text'][1].to(self.device)

            data['video'] = data['video'].to(self.device)
            data['video_mask'] = data['video_mask'].to(self.device)
            # data['video'] = data['video'].cuda(self.local_rank)
            # data['video_mask'] = data['video_mask'].cuda(self.local_rank)

            result = self.model(data)
            text_features = result['text_embeds']
            video_features_pooled_text = result['video_features_pooled_text']
            logits =result['logits']

            output_text = sim_matrix_training(text_features, video_features_pooled_text, self.pooling_type)

            output = (output_text + logits ) * 0.5

            loss = self.loss(output, self.model.clip.logit_scale)

            # torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            # loss /= dist.get_world_size()       # 对损失进行平均，可选步骤
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))   # 将值最大限制在max内

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:   # 每10轮打印一次
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        num_steps-1,
                        loss.detach().item()))

            if batch_idx in eval_steps:
                # del output,output_text,logits
                # torch.cuda.empty_cache()
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()
                
                # if dist.get_rank() == 0:
                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    # self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']
                    self._save_checkpoint(epoch, save_best=True)
                # if dist.get_rank() == 0:
                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self,epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        text_embed_arr_mean = []
        word_embed_arr = []
        farm_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        video_mean_arr = []




        with torch.no_grad():
            validloader = self.valid_data_loader
            # validloader.sampler.set_epoch(epoch)
            for _, data in tqdm(enumerate(validloader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                    # data['text'] = data['text'].cuda(self.local_rank)
                # else:
                    # data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)
                data['video_mask'] = data['video_mask'].to(self.device)
                # data['video'] = data['video'].cuda(self.local_rank)
                # data['video_mask'] = data['video_mask'].cuda(self.local_rank)
                
                result = self.model(data, return_all_frames=True)
                text_features = result['text_embeds']
                video_features_pooled_text = result['video_features_pooled_text']
                text_features_mean = result['text_embeds_mean']             #句子 计算句子和视频
                word_features = result['word_embeds']                       # 单词 计算单词和视频帧
                farm_features = result['video_frame_proto']                 # 视频帧  计算单词和视频帧
                video_features = result['video_features']                   
                video_features_mean = result['video_mean']                  # 视频 计算句子和视频
                logits =result['logits']


                video_mean_arr.append(video_features_mean.cpu())
                farm_embed_arr.append(farm_features.cpu())
                text_embed_arr.append(text_features.cpu())
                text_embed_arr_mean.append(text_features_mean.cpu())    
                word_embed_arr.append(word_features.cpu())
                vid_embed_arr.append(video_features.cpu())


                sims_batch_text = sim_matrix_training(text_features, video_features_pooled_text, self.pooling_type)

                sims_batch = (sims_batch_text + logits) * 0.5

                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)

                # torch.distributed.all_reduce(curr_loss, op=torch.distributed.ReduceOp.SUM)
                # curr_loss /= dist.get_world_size()  # 对损失进行平均，可选步骤


                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)
                
            text_embeds = torch.cat(text_embed_arr)
            text_embeds_mean = torch.cat(text_embed_arr_mean)
            vid_embeds = torch.cat(vid_embed_arr)
            word_embeds = torch.cat(word_embed_arr)
            video_mean_embeds = torch.cat(video_mean_arr)
            farm_embeds = torch.cat(farm_embed_arr)





            # Since we have all pairs, remove duplicate videos when there's multiple captions per video

            vid_embeds_per_video_id = {}
            vid_embeds_per_video_id_video_mean = {}
            vid_embeds_per_video_id_video_farm = {}

            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id_video_mean:
                    vid_embeds_per_video_id_video_mean[v_id] = video_mean_embeds[idx]

            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id_video_farm:
                    vid_embeds_per_video_id_video_farm[v_id] = farm_embeds[idx]

            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            video_mean_embeds = torch.stack([vid_embeds_per_video_id_video_mean[v_id] for v_id in vid_embeds_per_video_id_video_mean])

            farm_embeds = torch.stack([vid_embeds_per_video_id_video_farm[v_id] for v_id in vid_embeds_per_video_id_video_farm])

            
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()

            # text_embeds1 = generate_embeds_per_video_id3(text_embeds1, 
            #          all_vid_ids)
            # text_embeds1 = text_embeds1.sum(dim=1)

            # text_embeds = text_embeds.cuda()
            # vid_embeds = vid_embeds.cuda()

            vid_embeds_pooled_text = self.model.pool_frames(text_embeds, vid_embeds)

            # vid_embeds_pooled_text = []
            # for idx, v_id in enumerate(all_vid_ids):
            #     text_embeds1 = text_embeds[idx].unsqueeze(0)
            #     vid_embeds_pooled_text1 = self.model.pool_frames(text_embeds1, vid_embeds)
            #     vid_embeds_pooled_text.append(vid_embeds_pooled_text1)
            # vid_embeds_pooled_text = torch.cat(vid_embeds_pooled_text,dim=1)

            """MSVD"""
            # vid_embeds_pooled_text1 = []
            # vid_embeds_pooled_text2 = []
            # for idx , v_id in enumerate(all_vid_ids):
            #     text_embeds1 = text_embeds[idx].unsqueeze(0)
            #     for i in range(vid_embeds.shape[0]):
            #         vid_embeds1 = vid_embeds[i].unsqueeze(0)
            #         vid_embeds_pooled_text11 = self.model.pool_frames(text_embeds1, vid_embeds1)
            #         vid_embeds_pooled_text2.append(vid_embeds_pooled_text11)
            #     vid_embeds_pooled_text3 = torch.cat(vid_embeds_pooled_text2,dim=0)
            #     vid_embeds_pooled_text2 = []
            #     vid_embeds_pooled_text1.append(vid_embeds_pooled_text3)

            # vid_embeds_pooled_text = torch.cat(vid_embeds_pooled_text1,dim=1)



            """视频-句子"""
            text_embeds_mean = text_embeds_mean / text_embeds_mean.norm(dim=-1,keepdim=True)
            video_mean_embeds = video_mean_embeds / video_mean_embeds.norm(dim=-1,keepdim=True)
            video_text_logits = self.model.Logit_text_video(text_embeds_mean,video_mean_embeds)
            
            # text_embeds_mean, video_mean_embeds1 = generate_embeds_per_video_id1(text_embeds_mean, 
            #         video_mean_embeds, all_vid_ids, self.pooling_type)

            """MSVD"""
            # text_embeds_mean1 = text_embeds_mean
            # text_embeds_mean = text_embeds_mean.view(text_embeds_mean.shape[0]*text_embeds_mean.shape[1],text_embeds_mean.shape[2])
            # text_embeds_mean = text_embeds_mean / text_embeds_mean.norm(dim=-1,keepdim=True)
            # video_mean_embeds = video_mean_embeds / video_mean_embeds.norm(dim=-1,keepdim=True)
            # video_text_logits = []
            # for i in range(text_embeds_mean.shape[0]):
            #     text_embeds_mean2 = text_embeds_mean[i].unsqueeze(0)
            #     video_text_logits1 = self.model.Logit_text_video(text_embeds_mean2,video_mean_embeds)
            #     video_text_logits.append(video_text_logits1)
            # # video_text_logits = self.model.Logit_text_video(text_embeds_mean,video_mean_embeds)
            # video_text_logits = torch.cat(video_text_logits)
            # video_text_logits = video_text_logits.view(text_embeds_mean1.shape[0],text_embeds_mean1.shape[1],video_text_logits.shape[-1])



            """帧-单词"""
            word_embeds = F.normalize(word_embeds, p=2, dim=-1)
            farm_embeds = F.normalize(farm_embeds, p=2, dim=-1)
            word_frames_logits = self.model._attenion_over_fine_grained_sim_matrix(word_embeds,farm_embeds)

            """msvd"""
            # word_embeds, farm_embeds1 = generate_embeds_per_video_id2(word_embeds, 
            #         farm_embeds, all_vid_ids, self.pooling_type)
            # word_embeds1= word_embeds
            # word_embeds = word_embeds.view(word_embeds.shape[0]*word_embeds.shape[1],word_embeds.shape[2],word_embeds.shape[3])
            # word_embeds = F.normalize(word_embeds, p=2, dim=-1)
            # farm_embeds = F.normalize(farm_embeds, p=2, dim=-1)
            # word_frames_logits = []
            # for i in range(word_embeds.shape[0]):
            #     word_embeds3 = word_embeds[i].unsqueeze(0)
            #     word_frames_logits1 = self.model._attenion_over_fine_grained_sim_matrix(word_embeds3,farm_embeds)
            #     word_frames_logits.append(word_frames_logits1)
            # word_frames_logits = torch.cat(word_frames_logits)
            # word_frames_logits = word_frames_logits.view(word_embeds1.shape[0],word_embeds1.shape[1],word_frames_logits.shape[-1])


            logits_all = (video_text_logits + word_frames_logits ) * 0.5


            device = torch.device('cpu')
            # vid_embeds_pooled_text = vid_embeds_pooled_text.to(device)

            logits_all = logits_all.to(device)
            a , b = logits_all.shape
            logits_all = logits_all.view(a,1,b)

            self.model.pool_frames.cuda()
            self.model._attenion_over_fine_grained_sim_matrix.cuda()
            text_embeds_per_video_id, vid_embeds_pooled_per_video_id_text = generate_embeds_per_video_id(text_embeds, 
                    vid_embeds_pooled_text, all_vid_ids, self.pooling_type)

            # text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1,keepdim=True)
            # vid_embeds_pooled_per_video_id_text = vid_embeds_pooled_per_video_id_text / vid_embeds_pooled_per_video_id_text.norm(dim=-1,keepdim=True)
            """句子-帧"""
            # sims_text = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id_text, self.pooling_type)
            sims_text = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id_text, self.pooling_type)


            sims = (sims_text + logits_all ) * 0.5

            total_val_loss = total_val_loss / len(self.valid_data_loader)


            metrics = self.metrics
            res = metrics(sims)


            print(res)



            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)
            del sims
            torch.cuda.empty_cache()
            return res
