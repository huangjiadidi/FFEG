# Load the dataset
import torch.utils.data as data
from glob import glob
import random
from origin.data.helper import *
from config.config import opt
from util.util import *
import cv2
import PIL
from PIL import Image


class VoxCeleb(data.Dataset):
    def __init__(self, fraction=1):
        super().__init__()
        self.directories = glob('../voxceleb2/train/**/**')
        self.person_list = glob('../voxceleb2/train/**')
        self.person_dict = {name: index for index, name in enumerate(self.person_list) }
        

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        while True:
            try:
                video_path = random.choice(glob(self.directories[index] + '/*.mp4'))
                seq = get_frames_and_landmarks(video_path, opt.k + 1, generate_landmark=False)
                
                source = torch.stack(seq[:opt.k])
                target = seq[opt.k]

                target_mask = get_mask(target)
                
                pose = [get_landmark(x, from_tensor=True) for x in seq]
                source_pose = torch.stack(pose[:opt.k])
                target_pose = pose[opt.k]

                same_video_frame = get_frames_and_landmarks(video_path, 1, generate_landmark=False)[0]
                same_person_pose = get_landmark(same_video_frame, from_tensor=True)

                different_video_path = self.get_different_person_path(index)
                other = get_frames_and_landmarks(different_video_path, 1, generate_landmark=False)[0]
                other_pose = get_landmark(other, from_tensor=True)

                # same_video_path = self.get_same_person_path(index)
                # same_person = get_frames_and_landmarks(same_video_path, 1, generate_landmark=False)[0]
                # same_person_pose = get_landmark(other, from_tensor=True)

                return source, target, other, source_pose, target_pose, other_pose, same_person_pose, target_mask, index
            except:
                # write_loss('index error: ' + str(index), opt.error_file)
                index  = (index+1) % len(self.directories)

    def get_different_person_path(self, index):
        path = self.directories[index]
        name = '/'.join(path.split('/')[:4])
        name_index = self.person_dict[name]
        r = list(range(0,name_index)) + list(range(name_index+1, len(self.person_list)))
        different_person = random.choice(r)
        video_path = glob(self.person_list[different_person] + '/**/*.mp4')
        return  random.choice(video_path)


    def get_same_person_path(self, index):
        path = self.directories[index]
        name = '/'.join(path.split('/')[:4])
        name_index = self.person_dict[name]
        video_path = glob(self.person_list[name_index] + '/**/*.mp4')
        return  random.choice(video_path)



class VoxCelebTest(data.Dataset):
    def __init__(self, hold_number, max_frames):
        super().__init__()
        self.directories = glob('../voxceleb2/test/**')[:50]
        self.hold_number = hold_number
        self.max_frames = max_frames

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        try_times = 0
        i = 0
        while True:
            video_path = str(np.random.choice(glob(self.directories[index] + '/**/*.mp4'), 1)[0])
            hold, test = test_frame_and_landmarks(video_path, self.hold_number, max_frames=self.max_frames)
            other = other_video_frame_and_landmarks(self.get_different_person_path(index), max_frames=self.max_frames)

            if len(hold) < self.hold_number or len(test) < self.max_frames or len(other) < self.max_frames:
                try_times += 1
                if try_times > 10:
                    write_loss('index error: ' + str(index), opt.error_file)
                    index  = (index+1) % len(self.directories)
                    try_times = 0
                continue
            
            hold_face = torch.stack([x[0] for x in hold])
            hold_pose = torch.stack([x[1] for x in hold])
            test_face = torch.stack([x[0] for x in test])
            test_pose = torch.stack([x[1] for x in test])
            other_face = torch.stack([x[0] for x in other])
            other_pose = torch.stack([x[1] for x in other])

            return hold_face, hold_pose, test_face, test_pose, other_face, other_pose

    def get_different_person_path(self, index):
        different_video = (index + 1) % len(self.directories)
        video_path = str(np.random.choice(glob(self.directories[different_video] + '/**/*.mp4'), 1)[0])
        return  video_path







class VoxCeleb1(data.Dataset):
    def __init__(self):
        super().__init__()

        self.directories = glob('../voxceleb1/train/**/**/**')
        self.name_list = glob('../voxceleb1/train/**/**')
        self.name_dict = {name: index for index, name in enumerate(self.name_list) }
        
    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        while True:
            try:
                path = glob(self.directories[index] +  '/**')
                seq = get_frame_and_landmarks_from_folder(path, opt.k + 1)

                source = torch.stack(seq[:opt.k])
                target = seq[opt.k]

                target_mask = get_mask(target)

                source_pose = torch.stack([get_landmark(x, from_tensor=True) for x in seq[:opt.k]])
                target_pose = get_landmark(seq[opt.k], from_tensor=True, draw_head=True)

                # same_person_path = self.get_same_person_path(index)
                same_video_frame = get_frame_and_landmarks_from_folder(path, 1)[0]
                same_person_pose = get_landmark(same_video_frame, from_tensor=True, draw_head=True)

                different_person_path = self.get_different_person_path(index)
                other = get_frame_and_landmarks_from_folder(different_person_path, 1)[0]
                other_pose = get_landmark(other, from_tensor=True, draw_head=True)

                # same_person_path = self.get_same_person_path(index)
                # same_person = get_frame_and_landmarks_from_folder(same_person_path, 1, generate_landmark=False)[0]
                # same_person_pose = get_landmark(other, from_tensor=True)

                return source, target, other, source_pose, target_pose, other_pose, same_person_pose, target_mask, index

            except:
                # write_loss('index error: ' + str(index), opt.error_file)
                index  = (index+1) % len(self.directories)


    # ../voxceleb1/train/Kelly_Brook/1.6/4LcjCwYO6ck
    def get_different_person_path(self, index):
        path = self.directories[index]
        name = '/'.join(path.split('/')[:5])
        name_index = self.name_dict[name]
        r = list(range(0,name_index)) + list(range(name_index+1, len(self.name_list)))
        different_person_index = random.choice(r)
        image_path = glob(self.name_list[different_person_index] + '/**/**')
        return image_path

    def get_same_person_path(self, index):
        path = self.directories[index]
        name = '/'.join(path.split('/')[:5])
        name_index = self.name_dict[name]
        image_path = glob(self.name_list[name_index] + '/**/**')
        return image_path



class VoxCeleb1Test(data.Dataset):
    def __init__(self, hold_number, max_frames):
        super().__init__()
        names = glob('../voxceleb1/copy/**')[:100]
        self.directories = [ glob(name + '/1.6/**')[0] for name in names ]
        self.hold_number = hold_number
        self.max_frames = max_frames

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        path = glob(self.directories[index] + '/**')
        hold, test = test_folder_frames_and_landmarks(path, self.hold_number, self.max_frames)
        other = test_folder_frames_and_landmarks(self.get_different_person_path(index), self.hold_number, 0)[0]

        hold_face = torch.stack([x[0] for x in hold])
        hold_pose = torch.stack([x[1] for x in hold])
        test_face = torch.stack([x[0] for x in test])
        test_pose = torch.stack([x[1] for x in test])
        other_face = torch.stack([x[0] for x in other])
        other_pose = torch.stack([x[1] for x in other])
        return hold_face, hold_pose, test_face, test_pose, other_face, other_pose
        

    def get_different_person_path(self, index):
        different_video = (index + 1) % len(self.directories)
        return glob(self.directories[different_video] + '/**')






class VoxCeleb1Test(data.Dataset):
    def __init__(self, hold_number, max_frames):
        super().__init__()
        names = glob('../voxceleb1/copy/**')[:100]
        self.directories = [ glob(name + '/1.6/**')[0] for name in names ]
        self.hold_number = hold_number
        self.max_frames = max_frames

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        path = glob(self.directories[index] + '/**')
        hold, test = test_folder_frames_and_landmarks(path, self.hold_number, self.max_frames)
        other = test_folder_frames_and_landmarks(self.get_different_person_path(index), self.hold_number, 0)[0]

        hold_face = torch.stack([x[0] for x in hold])
        hold_pose = torch.stack([x[1] for x in hold])
        test_face = torch.stack([x[0] for x in test])
        test_pose = torch.stack([x[1] for x in test])
        other_face = torch.stack([x[0] for x in other])
        other_pose = torch.stack([x[1] for x in other])
        return hold_face, hold_pose, test_face, test_pose, other_face, other_pose
        

    def get_different_person_path(self, index):
        different_video = (index + 1) % len(self.directories)
        return glob(self.directories[different_video] + '/**')





class LivenessData(data.Dataset):
    def __init__(self, hold_number=32):
        super().__init__()
        from origin.data.list import videos
        self.videos = [ '../voxceleb2/' + '/'.join(video.split('/')[1:])  for video in videos]
        # self.videos = glob('../voxceleb2/copy/**/**/*.mp4')
        self.hold_number = hold_number
    
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        hold, _ = test_frame_and_landmarks(self.videos[index], self.hold_number, 0)
        hold_face = torch.stack([x[0] for x in hold])
        hold_pose = torch.stack([x[1] for x in hold])

        mouth = get_liveness_pose('mouth')
        nod = get_liveness_pose('nod')
        silent = get_liveness_pose('silent')
        yaw = get_liveness_pose('yaw')
        blink = get_liveness_pose('blink')
        
        return hold_face, hold_pose, mouth, nod, silent, yaw, blink

