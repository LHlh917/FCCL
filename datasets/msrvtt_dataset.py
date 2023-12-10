import os
import cv2
import sys
sys.path.append('/opt/data/private/LH/xpool')
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
# from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
import os
# import nltk
# from nltk.stem import WordNetLemmatizer
# from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP(r'/opt/data/private/LH/xpool/stanford-corenlp-4.5.4')

class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file = '/opt/data/private/LH/LH-v2t/xpool1/data/MSRVTT/MSRVTT_data.json'
        test_csv = '/opt/data/private/LH/LH-v2t/xpool1/data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        if config.msrvtt_train_file == '7k':
            train_csv = '/opt/data/private/LH/LH-v2t/xpool1/data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = '/opt/data/private/LH/LH-v2t/xpool1/data/MSRVTT/MSRVTT_train.9k.csv'

        def load_json(filename):
            with open(filename, "r") as f:
                return json.load(f)

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

            
    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        # if self.split_type == 'train': 
        #     # word = self._get_vidpath_and_nouns_by_index(video_id,caption)
        #     word = self._get_vidpath_and_nouns_by_index(video_id,caption)
        #     # pos_tag,_ = self.pos_tag(caption)
        #     # caption_q, words = self.erase_phrase(caption, pos_tag)
        # else: 
        #     word = self._get_vidpath_and_nouns_by_index(video_id,caption)
            # caption_q, words = self.erase_phrase(caption, pos_tag)
        # video_path, verbs_nouns, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs,video_mask = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
            # 'word': word,
            'video_mask' : video_mask
            # 'text': verbs_nouns,
        }

    
    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            # vid, verbs_nouns = self.all_train_pairs[index]
            video_path = os.path.join("/opt/data/private/LH/LH/xpool1/MSRVTT/videos/all", vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join("/opt/data/private/LH/LH/xpool1/MSRVTT/videos/all", vid + '.mp4')
            caption = self.test_df.iloc[index].sentence
            # verbs_nouns = self.test_df.iloc[index].sentence

        return video_path, caption, vid    #返回视频路径，标题，视频的ID
        # return video_path, verbs_nouns, vid
    
    # def _get_vidpath_and_nouns_by_index(self, video_id, caption):
    #     # from stanfordcorenlp import StanfordCoreNLP
    #     # nlp = StanfordCoreNLP(r'/opt/data/private/LH/xpool/stanford-corenlp-4.5.4',port=65534)
    #     # 初始化NLTK词性还原器
    #     lemmatizer = WordNetLemmatizer()
    #     # stanford_pos_tags = nlp.pos_tag(caption)
    #     words = nltk.word_tokenize(caption)
    #     pos_tags = nltk.pos_tag(words)
    #     words = []
    #     for word, pos in pos_tags:
    #         if pos.startswith('NN'):
    #             words.append(word)
    #         elif pos.startswith('VB') and word.lower() not in {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being'}:
    #             lemma = lemmatizer.lemmatize(word, pos='v')
    #             words.append(lemma)

            # elif pos.startswith('VB') and word.lower() not in {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being'}:
            #     lemma = lemmatizer.lemmatize(word, pos='v')  # 动词词性还原
            #     words.append(lemma)
        

        # 使用set去除重复的名词和动词
        unique_word = list(set(words))

        # 合并名词和动词的列表，以逗号分隔
        word = ', '.join(unique_word)
        # nlp.close()
        return word 



    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])    
    
    

            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            # print(annotation)
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)   #将所有对应这个视频的文本映射到这个列表中，即一个视频对应多个注释
    
    # def pos_tag(text):
    #     tokens = nltk.word_tokenize(text)
    #     return nltk.pos_tag(tokens)
            
    def erase_phrase(self,caption, pos_tag):
        words = []
        erased_text = caption
        if np.random.rand() > 0.5:
            tag_prefix = 'NN'  # 默认为名词短语
        else:
            tag_prefix = 'VB'  # 如果不是名词短语，则为动词短语

        phrases = [phrase for word, tag in pos_tag if tag.startswith(tag_prefix) for phrase in word.split()]
        # words = words.append(phrases)
        if phrases:
            phrase = random.choice(phrases)
            erased_text = erased_text.replace(phrase, '?')
            words.append(phrase)
        words = ', '.join(words)
        return erased_text, words
    
    