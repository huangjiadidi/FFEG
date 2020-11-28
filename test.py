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
hold_number = 8

# the maximum number of frame for testing
max_frames = 32

# video_path = str(np.random.choice(glob(self.directories[index] + '/**/*.mp4'), 1)[0])
video_path = "xxx"
other_video_path = "xxxx"
hold, test = test_frame_and_landmarks(video_path, hold_number, max_frames)
other, other_test = test_frame_and_landmarks(other_video_path, hold_number, max_frames=max_frames)

if len(hold) < hold_number or len(test) < max_frames or len(other) < max_frames:
    print("error orrur!")
    exit()

hold_face = torch.stack([x[0] for x in hold]).to(device)
hold_pose = torch.stack([x[1] for x in hold]).to(device)
test_face = torch.stack([x[0] for x in test]).to(device)
test_pose = torch.stack([x[1] for x in test]).to(device)
other_face = torch.stack([x[0] for x in other]).to(device)
other_pose = torch.stack([x[1] for x in other]).to(device)
other_test_face = torch.stack([x[0] for x in other_test]).to(device)
other_test_pose = torch.stack([x[1] for x in other_test]).to(device)


model = Model().to(device)
# loading the model from the checkpoint
checkpoint_path = "xxxx"

checkpoint = torch.load(checkpoint_path)
model.pose_encoder.load_state_dict(checkpoint["pose_encoder"])
model.generator.load_state_dict(checkpoint["generator"])
# model.pose_dis.load_state_dict(checkpoint["pose_dis"])
if "dis" in checkpoint:
    model.dis.load_state_dict(checkpoint["dis"])
if "optim_D" in checkpoint:
    model.optim_D.load_state_dict(checkpoint["optim_D"])
model.optim_G.load_state_dict(checkpoint["optim_G"])



model.stage2(hold_face, other_face, hold_pose, other_pose, None, None, None, k=3, i=0,fine_tune=True)

imgs = torch.cat([test_face, test_pose], 1)

with torch.no_grad():
    pose_latent = model.pose_encoder(other_test_pose)

    fake_sample = model.generator(imgs, pose_latent, k=3)[-1]

result = torch.cat([fake_sample, other_test_face])
concat_images = make_grid(result, nrow=12)
save_image(concat_images, 'generated_results.jpg', padding=1, normalize=True, range=(-1, 1), pad_value=-1)

