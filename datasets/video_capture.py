import cv2
import random
import numpy as np
import torch


class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames 
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        video_mask = np.zeros((1, num_frames), dtype=np.int64)
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path                 # 判断视频是否打开
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 查看视频总帧数

        # get indexes of sampled frames  计算需要采样的帧数，如果总帧数小于需要采样的帧数，则采用总帧数
        acc_samples = min(num_frames, vlen)
        # 构建 acc_samples 个等间隔区间，每个区间的长度是 ceil(vlen / acc_samples)，其中 start、stop 参数分别代表区间起始点和终止点，函数 linspace() 生成一个数量为 acc_samples+1 的等间隔数列，astype(int) 将其类型转化为整型数组，得到 intervals 数组
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
            
    # 如果是随机采样（'rand'），则在每个区间中随机选择一帧；如果是均匀采样（'uniform'），则在每个区间中选择中间帧。
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        error_occurred = False

        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if index >= vlen:
                print(f"Invalid frame index: {index} for video: {video_path}")
                continue
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                #cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                print(f"Error processing video file: {video_path}, frame index: {index}")
                error_occurred = True
                break 
                # raise ValueError

        if error_occurred:
            cap.release()
            return None, None, None  # 遇到错误，跳过当前视频

        if len(frames) < num_frames:
            video_mask[:len(frames)] = [1] * len(frames)
        else:
            video_mask[:num_frames] = [1] * num_frames

 # 如果采样的帧数小于需要采样的帧数，则在最后一帧后追加克隆帧，直到填满指定数量要求为止
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()

        return frames, frame_idxs,video_mask
