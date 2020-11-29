import torch

import torch.utils.data as data
from glob import glob
import random
from origin.data.helper import *
from config.config import opt
from util.util import *
import cv2
import PIL
from PIL import Image
from models.model import *
from torchvision.utils import make_grid, save_image

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0") if cuda else torch.device("cpu")

# the number of frame for fine-tuning
hold_number = 32

# the maximum number of frame for testing
max_frames = 32


#####################################################################################################
class VoxCeleb1Test(data.Dataset):
    def __init__(self, hold_number, max_frames):
        super().__init__()
        names = glob('../origin/dataset/unzippedFaces/**')[:100]
        self.directories = [glob(name + '/1.6/**')[0] for name in names]
        self.hold_number = hold_number
        self.max_frames = max_frames

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):

        path = glob(self.directories[index] + '/**')
        hold, test = test_folder_frames_and_landmarks(path, 1, self.max_frames)
        other = test_folder_frames_and_landmarks(self.get_different_person_path(index), self.hold_number, 0)[0]

        hold_face = torch.stack([x[0] for x in hold])
        hold_pose = torch.stack([x[1] for x in hold])
        print("length of test", len(hold))
        test_face = torch.stack([x[0] for x in test])
        test_pose = torch.stack([x[1] for x in test])
        other_face = torch.stack([x[0] for x in other])
        other_pose = torch.stack([x[1] for x in other])
        print(11111)
        return hold_face, hold_pose, test_face, test_pose, other_face, other_pose

    def get_different_person_path(self, index):
        different_video = (index + 1) % len(self.directories)
        return glob(self.directories[different_video] + '/**')


#############################################################################################################
# names = glob('../origin/dataset/unzippedFaces/**')[:100]
# print(len(names))
#
# directories = [glob(name + '/1.6/**')[0] for name in names]
# print(len(directories))
# path1 = glob(directories[i] + '/**' for i in range(6))
# path2 = glob(directories[i%len(directories)] + '/**' for i in range(6))
# hold, test = test_folder_frames_and_landmarks(path1, hold_number, max_frames)
# other, other_test = test_folder_frames_and_landmarks(path2, hold_number, max_frames)
#
# # if len(hold) < hold_number or len(test) < max_frames or len(other) < max_frames:
# #     print("error orrur!")
# #     exit()
#
# hold_face = torch.stack([x[0] for x in hold]).to(device)
# hold_pose = torch.stack([x[1] for x in hold]).to(device)
# test_face = torch.stack([x[0] for x in test]).to(device)
# test_pose = torch.stack([x[1] for x in test]).to(device)
# other_face = torch.stack([x[0] for x in other]).to(device)
# other_pose = torch.stack([x[1] for x in other]).to(device)
# other_test_face = torch.stack([x[0] for x in other_test]).to(device)
# other_test_pose = torch.stack([x[1] for x in other_test]).to(device)

#############################################################################################################
train_set = VoxCeleb1Test(hold_number, max_frames)
print(len(train_set))
training_data_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=2, batch_size=6, shuffle=False)

convertTensor = lambda batch : [x.to(device) for x in batch]
hold_face,  hold_pose, test_face, test_pose, other_face, other_pose = next(iter(training_data_loader))
all = [hold_face,  hold_pose, test_face, test_pose, other_face, other_pose]
hold_face,  hold_pose, test_face, test_pose, other_face, other_pose = convertTensor(all)

model = Model().to(device)
# loading the model from the checkpoint
checkpoint_path = "../origin/1-23000.pth"

checkpoint = torch.load(checkpoint_path)
model.pose_encoder.load_state_dict(checkpoint["pose_encoder"])
model.generator.load_state_dict(checkpoint["generator"])
# model.pose_dis.load_state_dict(checkpoint["pose_dis"])
if "dis" in checkpoint:
    model.dis.load_state_dict(checkpoint["dis"])
if "optim_D" in checkpoint:
    model.optim_D.load_state_dict(checkpoint["optim_D"])
model.optim_G.load_state_dict(checkpoint["optim_G"])

model.stage2(hold_face, other_face, hold_pose, other_pose, None, None, None, k=3, i=0, fine_tune=True)

imgs = torch.cat([test_face, test_pose], 1)

with torch.no_grad():
    pose_latent = model.pose_encoder(test_pose)

    fake_sample = model.generator(imgs, pose_latent, k=3)[-1]

result = torch.cat([fake_sample, test_face])
concat_images = make_grid(result, nrow=12)
save_image(concat_images, 'generated_results.jpg', padding=1, normalize=True, range=(-1, 1), pad_value=-1)
