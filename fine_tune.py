import torch
import numpy as np
from torch.utils.data import DataLoader
import os, sys, random
from data.dataloader import *
from config.config import opt
from models.model import *
from torchvision.utils import make_grid, save_image
from util.util import *
import time


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0") if cuda else torch.device("cpu")

if not cuda:
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(opt)


create_folder(opt.checkpoints_folder)


convertTensor = lambda batch : [x.to(device) for x in batch]


model = Model().to(device)
model.init_stage2_optim()
print(sum(p.numel() for p in model.parameters()))

root_folder = './liveness'
create_folder(root_folder)


checkpoint = torch.load(get_latest_model('./checkpoints1/**'))

flag = False

input_size = hold_size = 32
batch_size = 8


train_set = LivenessData(input_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=1, shuffle=False)

for i, batch in enumerate(training_data_loader):

    current_folder = root_folder + '/' + str(i + 1)
    silent_folder = current_folder + '/silent'
    nod_folder = current_folder + '/nod'
    yaw_folder = current_folder + '/yaw'
    mouth_folder = current_folder + '/mouth'
    blink_folder = current_folder + '/blink'

    create_folder(current_folder)
    create_folder(silent_folder)
    create_folder(nod_folder)
    create_folder(yaw_folder)
    create_folder(mouth_folder)
    create_folder(blink_folder)

    model.pose_encoder.load_state_dict(checkpoint["pose_encoder"])
    if flag:
        checkpoint["generator"]["face_emb"] = torch.randn(1, 3584).to(device)
    else:
        flag = True
    model.generator.load_state_dict(checkpoint["generator"])
    model.dis.load_state_dict(checkpoint["dis"])
    model.optim_D.load_state_dict(checkpoint["optim_D"])
    model.optim_G.load_state_dict(checkpoint["optim_G"])
    model.train()

    hold_face, hold_pose, mouth, nod, silent, yaw, blink = convertTensor(batch)
    

    hold_face = hold_face.view(-1, 3, 256, 256)
    hold_pose = hold_pose.view(-1, 3, 256, 256)

    with torch.no_grad():
        face, pose = hold_face[list(range(input_size))], hold_pose[list(range(input_size))]
        input = torch.cat([face, pose], 1)
        face_latent = model.generator.encoder(input, input_size).detach()

    model.generator.init_finetune(face_latent)
    

    for e in range(80):

        target_index = torch.randperm(hold_size)[:batch_size]
        target_img, target_pose = hold_face[target_index], hold_pose[target_index]

        print(target_img.size())
        print(target_img.size())
        
        res = model.stage2(target_img, target_img, target_pose, target_pose, None, None, None, 0, input_size, fine_tune=True)

        if (e + 1) % 10 == 0:
            info = 'B: {}, E {}, fm: {:.4f} vgg: {:.4f} g256: {:.4f} g128: {:.4f} d256: {:.4f} d128: {:.4f} d256: {:.4f} d128: {:.4f} l1: {:.4f}'.format(
                    i, e+1, *res[:9]
            )
            write_loss(info, opt.loss_file)
    
    
    model.eval()
    input_img, input_pose = hold_face[1], hold_pose[1]

    save_image(
        make_grid(input_img, nrow=1),
        current_folder + '/input.jpg',
        padding=1, normalize=True, range=(-1, 1), pad_value=-1
    )

    for frame_index in range(50):
        mouth_pose = mouth[0][[frame_index]]
        yaw_pose = yaw[0][[frame_index]]
        nod_pose = nod[0][[frame_index]]
        silent_pose = silent[0][[frame_index]]
        blink_pose = blink[0][[frame_index]]

        mouth_result = model.inference(mouth_pose, hold_size)
        yaw_result = model.inference(yaw_pose, hold_size)
        nod_result = model.inference(nod_pose, hold_size)
        silent_result = model.inference(silent_pose, hold_size)
        blink_result = model.inference(blink_pose, hold_size)

        save_image(mouth_result, mouth_folder + '/' + str(frame_index+1) + '.jpg', normalize=True, range=(-1, 1))
        save_image(yaw_result, yaw_folder + '/' + str(frame_index+1) + '.jpg', normalize=True, range=(-1, 1))
        save_image(nod_result, nod_folder + '/' + str(frame_index+1) + '.jpg', normalize=True, range=(-1, 1))
        save_image(silent_result, silent_folder + '/' + str(frame_index+1) + '.jpg', normalize=True, range=(-1, 1))
        save_image(blink_result, blink_folder + '/' + str(frame_index+1) + '.jpg', normalize=True, range=(-1, 1))

