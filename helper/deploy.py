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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score

from helper.ultrasound_ops import BaseTransform
from helper.config import DATASET_NUMS, DATASET_LABEL_NAME, KLAC_CORRECTION, KLFE_CORRECTION, KLHC_CORRECTION


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
        os.makedirs(p)


def npy2txt(npy_dir):
    npy = np.load(npy_dir)
    txt_dir = npy_dir.replace('.npy', '.txt')
    with open(txt_dir, 'w+') as f:
        for n in tqdm(npy):
            f.write('{}\r\n'.format(n))


def display_verbose(verbose):
    # identity
    if verbose:
        return lambda a: a
    else:
        return tqdm.tqdm


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


def get_frame_prediction(d):
    """Get class name and index from the whole dictionary
    E.g.,
    '{'index': '...', 'prediction': [{'class': 'sp_sfb', 'position': ..., 'score': ...}]}], 'label': ...}'
    ==>  (dict) {'class': 'sp_sfb', 'position': ..., 'score': ...}
    """
    return d['prediction'][0]


def correct_label(record):
    """Correct some mistakes in the manual labels"""
    video_idx = get_class_index(record)  # e.g., 'KLAC0570'
    frame_idx = get_frame_index(record)  # e.g., '12'
    cls = video_idx[:4]  # e.g., 'KLAC'

    true_right = [c for c in eval('{}_CORRECTION'.format(cls)) if c.startswith(video_idx)]
    prob_right = ['{}_{}.jpg'.format(video_idx, i) for i in range(frame_idx - 6, frame_idx + 7)]

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
            tmp = get_frame_prediction(m)['class']
        except (TypeError, IndexError):
            tmp = 'none'
        if tmp in ['sp_qn', 'sp_gg', 'sp_sfb']:
            interest.append(m)
    np.save(pred_path.replace('deploy_', 'interest_'), interest)


def strict_judgment(interest_path, threshold=0.9):
    """Do strict judgement from all interested predictions.
    That is, find one frame with the highest score in a video
    and then compare with label.

    Argument:
        interest_path: (str) path to file includes list of all interested records (dict)
    Return:
        y_truth: (list) true value in {0, 1} represents false and true sample respectively
        y_score: (list) probability ∈ [0, 1] with the same length as y_truth
    """
    preds, pred_hits = [], []

    interest = np.load(interest_path)
    cls = get_class_index(interest[0])[:4]

    # split into separate videos
    for vn, frames in groupby(interest, key=lambda d: d['index'].split('/')[-2]):
        best = sorted(frames, key=lambda d: get_frame_prediction(d)['score'])[-1]
        # filter: the latter half videos are the real test set
        if get_video_index(best) > DATASET_NUMS[cls] // 2 and get_frame_prediction(best)['score'] > threshold:
            if best['label'] not in ['others', 'missed']:
                preds.append(best)

    cls_name = DATASET_LABEL_NAME[cls]
    y_truth = [int(p['label'] == cls_name or correct_label(p)) for p in preds]
    y_score = [get_frame_prediction(p)['score'] for p in preds]

    # update corrected label
    corrected = []
    for i, p in enumerate(preds):
        if y_truth[i]:
            p['label'] = cls_name
        corrected.append(p)
    np.save(interest_path.replace('interest_results', 'corrected_best'), corrected)

    return y_truth, y_score


def statistics(model, cls, threshold=0.5):
    """Compute statistic by frame"""
    with open('deployment/{}/statistics.txt'.format(model), 'a+') as f_stat:
        interest_path = 'deployment/{}/{}/interest_results.npy'.format(model, cls)
        y_truth, y_score = strict_judgment(interest_path)

        y_truth = np.array(y_truth)
        y_pred = np.array(y_score) > threshold

        TP = sum(y_truth * y_pred)
        pred_nums, true_nums = sum(y_pred), sum(y_truth)
        nums = 'TP: {}\t# Pred: {}\t# True: {}\t'.format(TP, pred_nums, true_nums)

        # precision = precision_score(y_truth, y_pred)
        # recall = recall_score(y_truth, y_pred)
        # f1 = f1_score(y_truth, y_pred)
        # evaluation = 'Precision: {:.5f}\r\nRecall: {:.5f}\r\nF1 score: {:.5f}'.format(precision, recall, f1)

        ap = average_precision_score(y_truth, y_score)
        evaluation = 'AP: {:.4f}'.format(ap)
        now = datetime.datetime.now()
        info = """===> {} @ {}\n{}\r\n{}""".format(cls, now, nums, evaluation)
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
        self.transform = BaseTransform(self.input_size[0])

    def __getitem__(self, index):
        return self.pull_image(index)  # im, (h, w)

    def __len__(self):
        return len(self.frame_paths)

    def pull_image(self, index):
        cur_path = self.frame_paths[index]
        frame = cv2.imread(cur_path, 0)[..., None]
        im = self.transform(frame)[0]
        im = torch.from_numpy(im).permute(2, 0, 1)  # [c, h, w]
        return im.float(), cur_path


if __name__ == '__main__':
    exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
    os.chdir(exp_root)

    # concise_label()
    # merge_pred_true(model='retina')
    statistics(model='retina/FPN50', cls='KLAC')
