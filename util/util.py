import os
from glob import glob

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_loss(info, file='loss.txt'):
    loss_file = open(file, 'a')
    loss_file.write(info + '\n')
    loss_file.close()

def get_latest_model(path):
    checkpoints = glob(path)
    if len(checkpoints) == 0:
        return None
    print(max(checkpoints, key=os.path.getctime))
    return max(checkpoints, key=os.path.getctime)

def delete_hald_models(path):
	checkpoints = glob(path)
	checkpoints.sort(key=os.path.getmtime)
	total = len(checkpoints)
	if total >= 4:
		half = checkpoints[: total // 2]
		for f in half:
			os.remove(f)

def delete_series_half_checkpoint(path_list):
	for p in path_list:
		delete_hald_models(p + '/*')



