import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class MultiHeadedAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)   # （512， 512）
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)   # 1， 512， 32

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)  # 32 ，12 ， 1， 512
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)   # 32，1，12，512

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)  #32，12，512
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)   # 32，12，1，512
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)          #32，1，512，12

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q  #32，1，12，32
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights  # 32，1，512，32
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)   #32，32，512

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention) # 32，32，512  包含视频和文本特征
        return o


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)    # embed_dim = 512
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)     
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)    
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()


    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)   # 这是一个初始化操作，使用单位矩阵（单位对角矩阵）来初始化权重参数 param。nn.init.eye_() 是 PyTorch 中的函数，用于将张量初始化为单位矩阵
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)   #输入文本特征
        video_embeds = self.layer_norm1(video_embeds)   # 输入视频特征

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)  # 32，32，512
        attn_out = self.layer_norm2(attn_out)   # 输入的视频与文本融合后的特征

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out


class Logit_train(nn.Module):
    def __init__(self, config: Config):
        super(Logit_train, self).__init__()
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(77), requires_grad=True)

    def forward(self,hidden,video_mean):
        video_word_logits =  torch.sum(torch.matmul(hidden, video_mean.t()) \
            * torch.matmul(torch.softmax(torch.matmul(hidden, video_mean.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)
        return video_word_logits
    
class Logit_text_video(nn.Module):
    def __init__(self, config: Config):
        super(Logit_text_video, self).__init__()
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)
        self.video_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)

    def forward(self,text,video_mean):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text = text.to(device)
        video_mean = video_mean.to(device)
        # if self.training:
        #     # 在训练状态下的处理
        #     global_mat_weight = self.global_mat_weight.requires_grad_(True)
        # else:
        #     # 在评估状态下的处理
        #     global_mat_weight = self.global_mat_weight.requires_grad_(False)

        # linear_layer = nn.Linear(text.shape[0], video_mean.shape[0]).to(device)
        # text = linear_layer(text.permute(1,0))
        # text = text.permute(0,1)

        video_tetx_logits =  torch.matmul(torch.matmul(text, self.global_mat_weight), torch.matmul(video_mean,self.video_mat_weight).t())
        return video_tetx_logits

class Logit_word_frames(nn.Module):
    def __init__(self, config: Config):
        super(Logit_word_frames, self).__init__()
        self.global_frame_weight = nn.parameter.Parameter(torch.eye(12), requires_grad=True)

    def forward(self,word,frames):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word = word.to(device)
        frames = frames.to(device)
        video_tetx_logits =  torch.sum(torch.matmul(word, frames.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(word, frames.permute(0, 2, 1)) / 1e-2, dim=-1), self.global_frame_weight), dim=-1).t()

        return video_tetx_logits

class Logit_val_textvideo(nn.Module):
    def __init__(self, config: Config):
        super(Logit_val_textvideo, self).__init__()
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)

    def forward(self,hidden,video_mean):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        hidden = hidden.to(device)
        video_mean = video_mean.to(device)
        video_word_logits =  torch.matmul(torch.matmul(hidden, self.word_logit_weight), video_mean.t())
        return video_word_logits
    
class Logit_val_wordfram(nn.Module):
    def __init__(self, config: Config):
        super(Logit_val_wordfram, self).__init__()
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(12), requires_grad=True)

    def forward(self,hidden,video_mean):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        hidden = hidden.to(device)
        video_mean = video_mean.to(device)
        video_word_logits =  torch.sum(torch.matmul(hidden, video_mean.t()) \
            * torch.matmul(torch.softmax(torch.matmul(hidden, video_mean.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)
        return video_word_logits
    

class _attenion_over_fine_grained_sim_matrix(nn.Module):
    def __init__(self, config: Config):
        super(_attenion_over_fine_grained_sim_matrix, self).__init__()
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)
        self.local_mat_weight1 = nn.parameter.Parameter(torch.eye(512), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(28), requires_grad=True)
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(40), requires_grad=True)
        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(40), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(28), requires_grad=True)

    def forward(self,hidden,video_features1):
        # if self.training:
        #     # 在训练状态下的处理
        #     local_mat_weight = self.local_mat_weight.requires_grad_(True)
        #     local_mat_weight1 = self.local_mat_weight1.requires_grad_(True)
        #     word_mat_weight = self.word_mat_weight.requires_grad_(True)
        #     word_mat_weight2 = self.word_mat_weight2.requires_grad_(True)
        #     frame_mat_weight = self.frame_mat_weight.requires_grad_(True)
        #     frame_mat_weight2 = self.frame_mat_weight2.requires_grad_(True)
        # else:
        #     # 在评估状态下的处理
        #     local_mat_weight = self.local_mat_weight.requires_grad_(False)
        #     local_mat_weight1 = self.local_mat_weight1.requires_grad_(False)
        #     word_mat_weight = self.word_mat_weight.requires_grad_(False)
        #     word_mat_weight2 = self.word_mat_weight2.requires_grad_(False)
        #     frame_mat_weight = self.frame_mat_weight.requires_grad_(False)
        #     frame_mat_weight2 = self.frame_mat_weight2.requires_grad_(False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        hidden = hidden.to(device)
        video_features1 = video_features1.to(device)
        bs_video, num_frames, dim_video = video_features1.shape
        bs_text, num_words, dim_text = hidden.shape
        # fine_grained_sim_scores = torch.matmul(torch.matmul(hidden.view(-1, dim_text), self.local_mat_weight), torch.matmul(video_features1.view(-1, dim_video),self.local_mat_weight1).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]
        fine_grained_sim_scores = torch.matmul(torch.matmul(hidden.reshape(-1, dim_text), self.local_mat_weight), torch.matmul(video_features1.reshape(-1, dim_video),self.local_mat_weight1).t()).reshape(bs_text, num_words, bs_video, num_frames)
        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2
    
class logit_word_video(nn.Module):
    def __init__(self, config: Config):
        super(logit_word_video, self).__init__()
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(77), requires_grad=True)

    def forward(self,hidden,video_mean):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        hidden = hidden.to(device)
        video_mean = video_mean.to(device)
        video_word_logits =  torch.sum(torch.matmul(hidden, video_mean.t()) \
            * torch.matmul(torch.softmax(torch.matmul(hidden, video_mean.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)
        return video_word_logits
    
class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.local_rank = args.local_rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.local_rank : ctx.batch_size * (ctx.local_rank + 1)],
            None,
        )
    