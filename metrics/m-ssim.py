from skimage.measure import compare_ssim
from scipy.misc import imread




root_folder = './vox_finetune_imgs'



for tune_size in [1, 8, 32]:
    current_folder = root_folder + '/' + str(tune_size)

    total = 0
    count = 0

    for i in range(1, 101):
        current_name = current_folder + '/' + str(i)
        
        print('current i: ', i)

        for frame in range(1, 33):
            generated_frame = current_name + '/self_result/' + str(frame) + '.jpg'
            target_frame = current_name + '/target/' + str(frame) + '.jpg'

            a = imread(generated_frame)
            b = imread(target_frame)

            ssim = compare_ssim(a, b, multichannel=True, data_range=255)

            total += ssim
            count += 1
        
        
        
    print('--------------------------------- ', tune_size, total/count)

print('done')


