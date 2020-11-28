# finished epoch 4


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

create_folder(opt.img_folder)
create_folder(opt.checkpoints_folder)


save_interval = 1000 if cuda else 1
display_interval = 50 if cuda else 1

convertTensor = lambda batch : [x.to(device) for x in batch]


model = Model().to(device)


model.init_stage2_optim()

if get_latest_model(opt.checkpoints_folder + '/*'):
    checkpoint = torch.load(get_latest_model(opt.checkpoints_folder + '/*'))
    model.pose_encoder.load_state_dict(checkpoint["pose_encoder"])
    model.generator.load_state_dict(checkpoint["generator"])
    # model.pose_dis.load_state_dict(checkpoint["pose_dis"])
    if "dis" in checkpoint:
        model.dis.load_state_dict(checkpoint["dis"])
    if "optim_D" in checkpoint:
        model.optim_D.load_state_dict(checkpoint["optim_D"])
    model.optim_G.load_state_dict(checkpoint["optim_G"])
    # model.optim_PD.load_state_dict(checkpoint["optim_PD"])


print(sum(p.numel() for p in model.parameters()))

num_workers = 18
batch_size = 6

train_set = VoxCeleb()
training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)


for e in range(150):

    # pose_gadv = pose_dadv = diff_video = same_video = 0
    # l1_loss = [0, 0, 0, 0, 0]

    loss = [0] * 10
    start = time.time()
    for i, batch in enumerate(training_data_loader):
        write_loss('epoch: ' + str(e)  + ' batch: ' + str(i) + ' time: ' + str(time.time() - start), opt.log_file)
        start = time.time()

        source, target, other, source_pose, target_pose, other_pose, same_person_pose, target_mask, index = convertTensor(batch)
        source = source.view(-1, 3, 256, 256)
        source_pose = source_pose.view(-1, 3, 256, 256)
        
        if i % save_interval == save_interval - 1:
            res = model.stage2(source, target, source_pose, target_pose, other_pose, same_person_pose, target_mask, index, opt.k, get_image=True)
        else:
            res = model.stage2(source, target, source_pose, target_pose, other_pose, same_person_pose, target_mask, index, opt.k)
        
        loss = [i + j for i, j in zip(loss, res[:10])]

        # pose_gadv += res[0]
        # pose_dadv += res[1]
        # diff_video += res[2]
        # same_video += res[3]
        # l1_loss = [ i + j for i, j in zip(l1_loss, res[4]) ]

        if i % display_interval == display_interval - 1:
            loss = [x / display_interval for x in loss]
            info = 'E {}, B {}, fm: {:.4f} vgg: {:.4f} g256: {:.4f} g128: {:.4f} d256: {:.4f} d128: {:.4f} d256: {:.4f} d128: {:.4f} pose_gadv: {:.4f} pose_dadv: {:.4f}'.format(
                    e+1, i+1,*loss
            )
            write_loss(info, opt.loss_file)
            loss = [0] * 10

            # l1_loss = [int(x*100000/display_interval)/100000 for x in l1_loss]
            # info = 'E {}, B {}, pose_gadv : {:.5f} pose_dadv : {:.5f} diff_video : {:.5f} same_video : {:.5f} l1_loss: '.format(
            #         e+1, i+1,
            #         pose_gadv/display_interval,
            #         pose_dadv/display_interval,
            #         diff_video/display_interval,
            #         same_video/display_interval
            # ) + str(l1_loss)
            # write_loss(info, opt.loss_file)
            # pose_gadv = pose_dadv = diff_video = same_video = 0
            # l1_loss = [0, 0, 0, 0, 0]
        

        if i % save_interval == save_interval - 1:

            name = str(e + 1) + '-' + str(i+1)

            result = torch.cat([source[::opt.k, ...], target, res[-2], other, res[-1], other_pose + res[-1] - 1])
            concat_images = make_grid(result, nrow=batch_size)
            save_image(concat_images, opt.img_folder + '/' + name + '.jpg', padding=1, normalize=True, range=(-1, 1), pad_value=-1)

            delete_series_half_checkpoint([ opt.checkpoints_folder ])

            torch.save({
                "pose_encoder": model.pose_encoder.state_dict(),
                "generator": model.generator.state_dict(),
                "pose_dis": model.pose_dis.state_dict(),
                "dis": model.dis.state_dict(),
                "optim_G" : model.optim_G.state_dict(),
                "optim_D" : model.optim_D.state_dict(),
                "optim_PD" : model.optim_PD.state_dict(),
            },  opt.checkpoints_folder + '/' + name + '.pth')

            write_loss('model saved', opt.loss_file)

