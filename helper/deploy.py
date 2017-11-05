import codecs
import os
import glob
import datetime
import numpy as np

import cv2
import torch
import torch.utils.data as data

from tqdm import tqdm
from itertools import groupby
from helper.config import DATASET_NUMS, KLAC_CORRECTION, KLFE_CORRECTION, KLHC_CORRECTION


# helper function
# ------------------------------------------------------
def label_map(line):
    if '非' in line:
        res = 'nsp_'
    elif '标准' in line:
        res = 'sp_'
    else:
        return 'others'

    if '上腹部' in line:
        res += 'sfb'
    elif '股骨' in line:
        res += 'gg'
    elif '丘脑' in line:
        res += 'qn'
    elif '侧脑室' in line:
        res += 'cns'
    elif '胆囊' in line:
        res += 'dn'
    elif '小脑' in line:
        res += 'xn'
    return res


def sort_helper(p):
    """'data/Deploy/KLAC/KLAC0003/KLAC0003_86.jpg'
    ==> 'data/Deploy/KLAC/KLAC0003/KLAC0003_0086.jpg'
    """
    prefix, old_name = p.split('_')
    new_name = old_name.zfill(8)
    return '_'.join([prefix, new_name])


def check_dir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def npy2txt(npy_dir):
    npy = np.load(npy_dir)
    txt_dir = npy_dir.replace('.npy', '.txt')
    with open(txt_dir, 'w+') as f:
        for n in tqdm(npy):
            f.write('{}\r\n'.format(n))


def get_class_index(d):
    """Get class name and index from the whole dictionary
       E.g.,
       '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
       ==>  (str) 'KLAC0570'
       """
    return d['index'].split('/')[-2]


def get_video_index(d):
    """Get video index from the whole dictionary
    E.g.,
    '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
    ==>  (int) 570
    """
    return int(get_class_index(d)[-4:])


def get_frame_index(d):
    """Get frame index from the whole dictionary
       E.g.,
       '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
       ==>  (int) 12
       """
    return int(d['index'].split('_')[-1][:-4])


def correct_label(record):
    """Correct some mistakes in the manual labels"""
    video_idx = get_class_index(record)  # e.g., 'KLAC0570'
    frame_idx = get_frame_index(record)  # e.g., '12'
    cls = video_idx[:4]  # e.g., 'KLAC'

    true_right = [c for c in eval('{}_CORRECTION'.format(cls)) if c.startswith(video_idx)]
    prob_right = ['{}_{}.jpg'.format(video_idx, i) for i in range(frame_idx - 7, frame_idx + 7)]  # TODO:narrow limits

    for tr in true_right:
        if tr in prob_right:
            return True
    return False


def concise_label(save_dir='data/Deploy/LABEL'):
    """Abbreviate Chinese annotation into the mark, which is the same as label."""

    for cls in ['KLAC', 'KLFE', 'KLHC']:
        concise_labels = []
        label_files = sorted(glob.glob('{}/{}/*'.format(save_dir, cls)))
        for label_file in tqdm(label_files):
            with codecs.open(label_file, encoding='gbk') as f:
                lines = f.readlines()
                for line in lines:
                    # index: 'data/Deploy/KLAC/KLAC0001/KLAC0001_1.jpg', keep same with deployed index
                    index = '{}/{}'.format(label_file.replace('LABEL/', '').rstrip('.txt'), line.split(' ')[0])
                    label = label_map(line)
                    concise_labels.append({'index': index, 'label': label})
        np.save(save_dir + '/{}_{}.npy'.format(cls, 'concise_labels'), concise_labels)


# ------------------------------------------------------
def merge_pred_true(model, cls):
    pred_path = 'deployment/{}/{}/deploy_results.npy'.format(model, cls)
    true_path = 'data/Deploy/LABEL/{}_concise_labels.npy'.format(cls)
    pred, true = [np.load(p) for p in [pred_path, true_path]]
    merged, i, j = [], 0, 0
    while i < len(pred) and j < len(true):
        p, t = pred[i], true[j]
        if p['index'] == t['index']:
            m = {**p, **t}
            merged.append(m)
            i += 1
            j += 1
        else:
            merged.append(p.update({'label': 'missed'}))
            i += 1

    # print('pred: {} | merged: {} | true: {}'.format(len(pred), len(merged), len(true)))
    assert len(pred) == len(merged), '#pred : {} is not equal with #merged{}'.format(len(pred), len(merged))
    np.save(pred_path.replace('deploy_', 'merged_'), merged)
    # filter the positive prediction
    interest = []
    for m in merged:
        try:
            tmp = m['prediction'][0]['class']
        except (TypeError, IndexError):
            tmp = 'none'
        if tmp in ['sp_qn', 'sp_gg', 'sp_sfb']:
            interest.append(m)
    np.save(pred_path.replace('deploy_', 'interest_'), interest)


def strict_judgment(interest_path):
    """Do strict judgement from all interested predictions.
    That is, find one frame with the highest score in a video
    and then compare with label.

    Argument:
        interest_path: (str) path to file includes list of all interested records (dict)
    Return:
        best_nums: (int) true positive numbers
        pred_nums: (int) predict positive numbers (exclude mislabeled samples)
        bests: (list) list of best predicted records.
    """
    pred_nums, best_nums, bests = 0, 0, []

    interest = np.load(interest_path)
    cls = get_class_index(interest[0])[:4]

    # split into separate videos
    for vn, frames in groupby(interest, key=lambda d: d['index'].split('/')[-2]):
        best = sorted(frames, key=lambda d: d['prediction'][0]['score'])[-1]
        # Filter: the latter half videos are the real test set
        if get_video_index(best) > DATASET_NUMS[cls] // 2:
            pred_nums += 1
            if best['prediction'][0]['class'] == best['label'] or correct_label(best):
                best_nums += 1
                bests.append(best)
            elif best['label'] in ['others', 'missed']:
                pred_nums -= 2

    best_path = interest_path.replace('interest', 'best')
    np.save(best_path, bests)
    return best_nums, pred_nums


def statistics(model, cls):
    """Compute statistic by frame"""
    with open('deployment/{}/statistics.txt'.format(model), 'a+') as f_stat:
        interest_path = 'deployment/{}/{}/interest_results.npy'.format(model, cls)
        best_nums, pred_nums = strict_judgment(interest_path)

        now = datetime.datetime.now()
        info = '{} precision: {}/{} {:.5f} @ {}\r\n'.format(cls, best_nums, pred_nums, best_nums / pred_nums, now)
        f_stat.write(info)
        print(info)


class DeployDataset(data.Dataset):
    def __init__(self, root_dir='data/Deploy/', cls='KLAC'):
        """
        Arguments:
            root_dir (string): Path to the image file.
            cls (string): 'KLHC' / 'KLFE' / 'KLAC'
        Returns:
            + image (tensor)
            + original image shape, before resize

        """
        self.root = os.path.join(root_dir, cls)
        self.cls = cls
        self.video_names = sorted([d for d in os.listdir(self.root) if not d.endswith('.wmv')])
        self.frame_paths = sorted(glob.glob('{}/*/*.jpg'.format(self.root)), key=sort_helper)
        self.image_size = cv2.imread(self.frame_paths[0]).shape[:-1]
        self.input_size = (300, 300)

    def __getitem__(self, index):
        return self.pull_image(index)  # im, (h, w)

    def __len__(self):
        return len(self.frame_paths)

    def pull_image(self, index):
        cur_path = self.frame_paths[index]
        frame = cv2.imread(cur_path)
        frame = frame[:, :, (2, 1, 0)]  # to rgb

        im = cv2.resize(frame, self.input_size).astype(np.float32)
        im -= (104.0, 117.0, 123.0)

        # [channel, h, w]
        return torch.from_numpy(im).permute(2, 0, 1), cur_path


if __name__ == '__main__':
    exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
    os.chdir(exp_root)

    # concise_label()
    # merge_pred_true(model='retina')
    statistics(model='retina/FPN50', cls='KLAC')
