"""
Apply algorithm in reality, that is, performing inference by video.
"""
import cv2
import os
import imageio
import numpy as np
import visdom

import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader

from glob import glob
from tqdm import tqdm
from itertools import groupby
from collections import Counter
from skimage.color import rgb2gray
from helper.config import VOC_CLASSES as labels
from helper.deploy import check_dir, get_frame_prediction
from helper.ultrasound_ops import normalize
from retina.encoder import DataEncoder
from retina.retinanet import RetinaNet

# Init
# -------------------------
EXP_ROOT = '/home/zengyu/Lab/pytorch/standard-panel-classification'
DATA_ROOT_DIR = 'data/NewDeploy'
os.chdir(EXP_ROOT)

base = 'FPN34'
checkpoint = torch.load('retina/checkpoints/{}/best_ckpt.pth'.format(base))
print('Loading {} model from epoch {}...'.format(base, checkpoint['epoch']))

net = RetinaNet(base)
net.load_state_dict(checkpoint['net'])
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
torch.cuda.set_device(3)
net.eval().cuda()

vis = visdom.Visdom()


def read_into_memory(videoname, size):
    video_reader = imageio.get_reader(videoname, 'ffmpeg')

    ar = []
    for im in video_reader:
        # 1. convert to grayscale
        im = rgb2gray(im)
        # 2. resize
        im = cv2.resize(im, size)
        # 3. put into list
        ar.append(normalize(im))

    return torch.from_numpy(np.array(ar, dtype=np.float32)).unsqueeze_(1), im.shape


def read_frames(videoname, index):
    video_reader = imageio.get_reader(videoname, 'ffmpeg')
    if not isinstance(index, (list, tuple)):
        index = [index]
    return np.array([im for i, im in enumerate(video_reader) if i in index])


def find_class(frames_array):
    classes = [get_frame_prediction(f)['class'].split('_')[-1] for f in frames_array]
    return Counter(classes).most_common(1)[0][0]


class PracticeDataset(data.Dataset):
    def __init__(self, videoname, input_size=(300, 300)):
        self.videoname = videoname
        self.input_size = input_size
        self.data_tensor, self.image_size = read_into_memory(videoname, input_size)

    def __getitem__(self, index):
        cur_path = self.videoname[:-4] + '_{}.jpg'.format(index)
        return self.data_tensor[index], cur_path

    def __len__(self):
        return self.data_tensor.size(0)


def do_save_deployment(dataset, save_dir, batch_size=2, visualize=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_num = len(dataset)
    h, w = dataset.input_size
    ori_h, ori_w = dataset.image_size
    encoder = DataEncoder()

    deploy_results = []
    for im_batch, p_batch in tqdm(dataloader, total=total_num // batch_size, unit=' frames({})'.format(batch_size)):
        x_batch = Variable(im_batch, volatile=True).cuda()
        loc_preds, cls_preds = net(x_batch)

        # scale each detection back up to the image
        scale = torch.Tensor([ori_w, ori_h, ori_w, ori_h]) / dataset.input_size[0]

        for i, (loc_pred, cls_pred) in enumerate(zip(loc_preds, cls_preds)):
            boxes, labels_idx, scores = encoder.decode(loc_pred.cpu().data.squeeze(),
                                                       cls_pred.cpu().data.squeeze(),
                                                       (w, h), return_score=True)
            pred = []
            for box, label_idx, score in zip(boxes, labels_idx, scores):
                pt = (box * scale).numpy().astype(int)
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                label_name = labels[label_idx]
                pred.append({'class': label_name, 'score': score, 'position': coords})
            deploy_results.append({'index': p_batch[i], 'prediction': pred})

            if visualize:
                vis.image(im_batch[i][0], win='input', opts=dict(title='input image', caption='300 x 300'))
    save_dir_with_cls = save_dir.replace('.npy', '_{}.npy'.format(find_class(deploy_results)))
    np.save(save_dir_with_cls, deploy_results)


def grab_frame(videoname, result_path, draw_boxes=False):
    """Grab the best frame and return the class of video, including 'KLAC', 'KLFE', 'KLHC'"""

    common = result_path[:-7] + '*'
    frames_records = np.load(glob(common)[0])
    frames = [f for f in frames_records if get_frame_prediction(f)['class'].startswith('sp_')]
    try:
        best_record = sorted(frames, key=lambda d: get_frame_prediction(d)['score'])[-1]
    except IndexError:
        return

    if get_frame_prediction(best_record)['score'] > 0.7:
        frame_cls = get_frame_prediction(best_record)['class'].split('_')[-1]
        frame_name = best_record['index']
        index = int(frame_name.split('_')[-1][:-4])
        savename = '{}/{}_{}.jpg'.format(os.path.dirname(result_path), os.path.basename(frame_name)[:-4], frame_cls)
        cv2.imwrite(savename, read_frames(videoname, index)[0])


def statistical_analysis(save_root_dir):
    def get_cls(path):
        return path.split('_')[-1][:-4]

    name_map = {'npy': 'true', 'jpg': 'pred'}
    for fmt in ['npy', 'jpg']:
        precess = name_map[fmt]
        print('{} Processing {}... {}'.format('*' * 15, precess.upper(), '*' * 15))

        lst = sorted(glob('{}/*/*/*.{}'.format(save_root_dir, fmt)), key=get_cls)
        items = [(key, list(group)) for key, group in groupby(lst, get_cls)]
        for cls, item_list in items:
            print(cls, len(item_list))
            eval(precess).update({cls: len(item_list)})


def main():
    videoname_list = glob('{}/*/*/*.avi'.format(DATA_ROOT_DIR))
    total_nums = len(videoname_list)
    save_root_dir = 'deployment/practice_res34'

    for i, videoname in enumerate(videoname_list):
        print('Processing {}'.format(videoname), '*' * 30, '{}/{}'.format(i, total_nums))
        save_path = videoname.replace(DATA_ROOT_DIR, save_root_dir)[:-4] + '.npy'
        check_dir(os.path.dirname(save_path))

        dataset = PracticeDataset(videoname)
        do_save_deployment(dataset, save_path, visualize=True)
        grab_frame(videoname, save_path, draw_boxes=False)


if __name__ == '__main__':
    main()
