import torch
import torch.nn as nn
from config.base_config import Config
from model.transformer import Transformer,Logit_train,Logit_val_textvideo,Logit_text_video,Logit_word_frames,Logit_val_wordfram,_attenion_over_fine_grained_sim_matrix
import json
from model.differential_topk import VisualTokenSelection
from model.clip_model import Extract,VisualTransformer, TransformerClip,_mean_pooling_for_similarity_visual,_mean_pooling_for_similarity_sequence
import torch.nn.functional as F
import torch.distributed as dist

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("/openaiclip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        self.logit_train = Logit_train(config)

        self.Logit_val_textvideo = Logit_val_textvideo(config)
        self.Logit_val_wordfram = Logit_val_wordfram(config)

        self.Logit_text_video = Logit_text_video(config)
        self.Logit_word_frames = Logit_word_frames(config)
        # self.logit_word_video = logit_word_video(config)
        self.transformerClip = TransformerClip(width=512, layers=4,heads=8)
        self.frame_position_embeddings = nn.Embedding(77, 512)
        self.logit_scale = self.clip.logit_scale.exp()

        self._attenion_over_fine_grained_sim_matrix = _attenion_over_fine_grained_sim_matrix(config)
        self.word_pro_num = 28
        # self.event_layer_num = 2
        # self.event_num = 12
        # self.frame_num = 12
        self.set_dim = 512
        # self.patch_num = 12
        # self.patch_prototype_weight = nn.Sequential(
        #     nn.Linear(self.set_dim, self.set_dim), nn.ReLU(inplace=True),
        #     nn.Linear(self.set_dim, self.patch_num-1), nn.ReLU(inplace=True))
        # self.max_frames = 12

        self.word_prototype_weight = nn.Sequential(
            nn.Linear(self.set_dim, self.set_dim), nn.ReLU(inplace=True),
            nn.Linear(self.set_dim, self.word_pro_num), nn.ReLU(inplace=True))

        self.visual_token_selector = VisualTokenSelection(10, 512, topk=3)


        self.extract = Extract()
        # self.extract1 = Extract1()

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text'][0].cuda()
        text_data_mask = data['text'][1].cuda()
        # word_data = data['word']
        video_data = data['video']
        video_mask = data['video_mask']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features,hidden = self.clip.encode_text(text_data,return_hidden = True)
            video_features, visual_pixel = self.clip.encode_image(video_data,return_hidden = True)
            bs_pair = video_mask.size(0)
            # visual_pixel = visual_pixel.reshape(bs_pair, -1, visual_pixel.size(-1))
            visual_pixel = visual_pixel.reshape(bs_pair, 12, -1, visual_pixel.size(-1))  # 重构为 [32, 12, 50, 512]
            visual_pixel = select_random_frames(visual_pixel, 10)
            visual_pixel = visual_pixel.reshape(bs_pair, -1, visual_pixel.size(-1))
            visual_pixel = self.visual_token_selector(visual_pixel)

            # bs_pair = text_features.size(0)
            text_feat = hidden.view(bs_pair, -1, hidden.size(-1)) #(128, 32, 512)
            """phrase"""
            # text_feat = self.extract(text_feat)
            """weight"""
            # word_weights = self.word_prototype_weight(text_feat) #(128, 32, 28)
            # text_word_proto = torch.einsum('bmd,bmn->bnd', text_feat, word_weights) #（128,28, 512）
            text_word_proto = text_feat
            
            video_features = video_features.reshape(batch_size, self.config.num_frames, -1)


            video_features_pooled_text = self.pool_frames(text_features, video_features) #这行代码的作用是将文本特征和视频特征进行融合。具体来说，它调用了 pool_frames 函数，将视频特征进行了池化（即将视频特征序列压缩为单个特征向量），然后将文本特征和池化后的视频特征按照某种方式进行了融合，得到了最终的特征向量


            """video"""
            video_features1 = video_features
            seq_length = video_features1.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_features1.device)
            position_ids = position_ids.unsqueeze(0).expand(video_features1.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_features1 = video_features1 + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_features1 = video_features1.permute(1, 0, 2)  # NLD -> LND
            video_features1 = self.transformerClip(video_features1, extended_video_mask)
            video_features1 = video_features1.permute(1, 0, 2)  # LND -> NLD
            video_features1 = video_features1 + video_features

            # video_features = video_features.reshape(-1,12,512)
            # video_features2 = video_features1
            video_features1 = video_features1 / video_features1.norm(dim=-1,keepdim=True)
            video_mean = _mean_pooling_for_similarity_visual(video_features1,video_mask)

            # 进行文本mean处理
            # text_word_proto = F.normalize(text_word_proto, p=2, dim=-1)
            # text_features_mean = _mean_pooling_for_similarity_sequence(text_word_proto,text_data_mask)

            video_features = video_features.contiguous()
            video_mean = video_mean.contiguous()
            text_features = text_features.contiguous()
            hidden = hidden.contiguous()
            visual_pixel = visual_pixel.contiguous()
            text_word_proto = text_word_proto.contiguous()

            video_mean = video_mean / video_mean.norm(dim=-1,keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)


        logit_scale = self.clip.logit_scale.exp()
        """text-video_mean"""
        video_text_logits =  self.Logit_text_video(text_features,video_mean)

        """word-frames"""

        text_word_proto = F.normalize(text_word_proto, p=2, dim=-1)
        visual_pixel = F.normalize(visual_pixel, p=2, dim=-1)
 

        word_frames_logits =  self._attenion_over_fine_grained_sim_matrix(text_word_proto,visual_pixel)



        logit = (video_text_logits + word_frames_logits ) * 0.5
        # logit =  video_text_logits 



        if return_all_frames:
            return {
                'text_embeds': text_features,
                'video_features_pooled_text': video_features_pooled_text,
                'text_embeds_mean': text_features,
                'word_embeds': text_word_proto,
                'video_features' : video_features,
                'video_mean' : video_mean,
                'logits' : logit,
                'logit_scale' : logit_scale,
                'video_frame_proto':visual_pixel
            }

        return {
        'text_embeds': text_features,
        'video_features_pooled_text': video_features_pooled_text,
        'logits' : logit,
        'logit_scale' : logit_scale
         }





def sinkhorn_knopp(log_sim_matrix, n_iters=4, detach=False):
    if detach:
        log_sim_matrix = log_sim_matrix.detach()
    # m= torch.max(log_sim_matrix)
    m = log_sim_matrix.max()
    _log_sim_matrix = log_sim_matrix - m
    sim_matrix = torch.exp(_log_sim_matrix)
    b = 1 / sim_matrix.sum(0)
    for _ in range(n_iters):
        a = 1 / (sim_matrix @ b)
        b = 1 / (a @ sim_matrix)

    log_a = a.log()
    log_b = b.log() - m

    return F.log_softmax(log_a, dim=0), F.log_softmax(log_b, dim=0)


def select_random_frames(x, num_frames_to_select):
    B, F, S, D = x.shape  # B: batch size, F: number of frames, S: number of segments, D: feature dimension
    selected_frames = []

    for i in range(B):
        # 为每个样本随机选择帧的索引
        selected_indices = torch.randperm(F)[:num_frames_to_select].sort()[0]
        selected_frames.append(x[i, selected_indices, :, :])

    # 将选定的帧重新组合成一个新的批次
    selected_frames = torch.stack(selected_frames)
    return selected_frames
