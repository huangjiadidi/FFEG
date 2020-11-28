import cv2
from glob import glob
from util.util import *


root_folder = './demo'
create_folder(root_folder)

for video in range(1, 51):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(root_folder + '/out' + str(video) + '.mp4', fourcc, 20.0, (512, 256))

    for i in range(1, 51):
        origin = cv2.imread('./finetune_imgs/8/' + str(video) + '/other/' + str(i) + '.jpg')
        result = cv2.imread('./finetune_imgs/8/' + str(video) + '/other_result/' + str(i) + '.jpg')
        img = cv2.hconcat([origin,result])
        out.write(img)

    out.release()



