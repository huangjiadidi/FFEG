import torch
import dlib

import cv2
import numpy as np
import PIL

from PIL import Image

from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, RandomHorizontalFlip, RandomRotation

import random

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cuda = torch.cuda.is_available()

def plot_landmark(landmarks, side_length=256, draw_head=True):
    import matplotlib
    if not cuda:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpi = 100
    fig = plt.figure(figsize=(side_length / dpi, side_length / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    plt.imshow(np.ones((side_length, side_length, 3)))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    if draw_head:
        ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords



def get_landmark(image, from_tensor=False, draw_head=True):
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    if from_tensor:
        image = (image + 1) / 2
        image = np.array(ToPILImage()(image))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 0)

    shape = predictor(gray, rects[0])
    pred = shape_to_np(shape)

    return transform(plot_landmark(pred, draw_head=draw_head))



def get_frames_and_landmarks(video_path, num_frames, generate_landmark=False):
    augumentation = Compose([RandomHorizontalFlip(0.5)])
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    res = []
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length < num_frames:
        random_frames = [random.choice(range(0, length - 1)) for _ in range(num_frames)]
    else:
        random_frames = random.sample(range(0, length - 1), num_frames)
    
    for frame_number in random_frames:
        cap.set(1, frame_number)
        _, frame = cap.read()
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = augumentation(Image.fromarray(frame, mode='RGB'))
        frame = Image.fromarray(frame, mode='RGB')
    
        if generate_landmark:
            landmark = get_landmark(frame)
            res.append([transform(frame), landmark])
        else:
            res.append(transform(frame))
    cap.release()
    return res

    

def get_rotate_video_frames(video_path, num_frames):
    augumentation = Compose([RandomHorizontalFlip(0.5), RandomRotation(30)])
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    res = []
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length < num_frames:
        random_frames = [random.choice(range(0, length - 1)) for _ in range(num_frames)]
    else:
        random_frames = random.sample(range(0, length - 1), num_frames)
    for frame_number in random_frames:
        cap.set(1, frame_number)
        _, frame = cap.read()
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = augumentation(Image.fromarray(frame, mode='RGB'))
        res.append(transform(frame))
    cap.release()
    return res




def test_frame_and_landmarks(video_path, hold_number=32, max_frames=32):
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    hold = []
    test = []

    hold_count = 0
    test_count = 0

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = ([i for i in range(length)])
    np.random.seed(0)
    np.random.shuffle(frame_index)

    for index in frame_index:
        cap.set(1, index)
        _, frame = cap.read()
        
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if hold_count < hold_number:
            try:
                landmark = get_landmark(frame)
                frame = Image.fromarray(frame, mode='RGB')
                hold.append([transform(frame), landmark])
                hold_count += 1
            except:
                pass
        elif test_count < max_frames:
            try:
                landmark = get_landmark(frame)
                frame = Image.fromarray(frame, mode='RGB')
                test.append([transform(frame), landmark])
                test_count += 1
            except:
                pass
        else:
            break
    return hold, test





def other_video_frame_and_landmarks(video_path, max_frames=32):
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    frame_count = 0
    frames = []

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = ([i for i in range(length)])
    np.random.seed(0)
    np.random.shuffle(frame_index)

    for index in frame_index:
        cap.set(1, index)
        _, frame = cap.read()
        
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            landmark = get_landmark(frame)
            frame = Image.fromarray(frame, mode='RGB')
            frames.append([transform(frame), landmark])
            frame_count += 1
        except:
            print('other test dataset error')
        
        if frame_count == max_frames:
            break

    return frames






def get_frame_and_landmarks_from_folder(path, num_frames, generate_landmark=False):
    augumentation = Compose([RandomHorizontalFlip(0.5)])
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    length = len(path)
    if len(path) > num_frames:
        random_frames = [random.choice(range(0, length - 1)) for _ in range(num_frames + 1)]
    else:
        random_frames = random.sample(range(0, length - 1), num_frames + 1)
    
    res = []
    for index in random_frames:
        frame = cv2.imread(path[index])
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = augumentation(Image.fromarray(frame, mode='RGB'))
        # frame = Image.fromarray(frame, mode='RGB')

        if generate_landmark:
            landmark = get_landmark(np.array(frame))
            res.append([transform(frame), landmark])
        else:
            res.append(transform(frame))
    return res


def get_rotate_frame_from_folder(path, num_frames):
    augumentation = Compose([RandomHorizontalFlip(0.5), RandomRotation(30)])
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    length = len(path)
    if len(path) > num_frames:
        random_frames = [random.choice(range(0, length - 1)) for _ in range(num_frames + 1)]
    else:
        random_frames = random.sample(range(0, length - 1), num_frames + 1)
    
    res = []
    for index in random_frames:
        frame = cv2.imread(path[index])
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = augumentation(Image.fromarray(frame, mode='RGB'))
        res.append(transform(frame))
    return res




def test_folder_frames_and_landmarks(folder_path, hold_number=32, max_frames=32):
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    hold = []
    test = []

    hold_count = 0
    test_count = 0

    for img_path in folder_path:
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmark = get_landmark(frame)
        frame = Image.fromarray(frame, mode='RGB')

        if hold_count < hold_number:
            hold.append([transform(frame), landmark])
            hold_count += 1
        elif test_count < max_frames:
            test.append([transform(frame), landmark])
            test_count += 1
        else:
            break

    return hold, test



def get_liveness_pose(motion):
    transform = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    video = []
    for i in range(1, 51):
        img = cv2.imread('./pose/' + motion + '/pose_' + str(i) + '.jpg')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.append(transform(img))
    return torch.stack(video)



def get_mask(face):
    face = (face + 1) / 2
    face = np.array(ToPILImage()(face))
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 0)
    shape = predictor(gray, rects[0])
    pred = shape_to_np(shape)
    x, y = pred[:, 0], pred[:, 1]
    # mask = np.array(list((zip(x[0:17], y[0:17]))) + list((zip(x[22:27][::-1], y[22:27][::-1]))) + list((zip(x[17:22][::-1], y[17:22][::-1]))), dtype=np.int32)
    left_eye = np.array(list((zip(x[36:42], y[36:42]))), dtype=np.int32)
    right_eye = np.array(list((zip(x[42:48], y[42:48]))), dtype=np.int32)
    mouth = np.array(list((zip(x[48:60], y[48:60]))), dtype=np.int32)
    landmark = np.zeros((256, 256, 3), np.float32)
    cv2.drawContours(landmark,[left_eye, right_eye, mouth],-1,(255,255,255),-1)
    landmark = landmark.astype(face.dtype)
    return ToTensor()(landmark)
