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
