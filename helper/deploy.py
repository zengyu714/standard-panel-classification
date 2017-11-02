import codecs
import os
import glob
import numpy as np

import cv2
import torch
import torch.utils.data as data
from tqdm import tqdm


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


def concise_label(save_dir='data/Deploy/LABEL'):
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        concise_labels = []
        label_files = sorted(glob.glob('{}/{}/*'.format(save_dir, cls)))
        for label_file in tqdm(label_files):
            with codecs.open(label_file, encoding='gbk') as f:
                lines = f.readlines()
                for line in lines:
                    # index: 'data/Deploy/KLAC/KLAC0001/KLAC0001_1.jpg'
                    index = '{}/{}'.format(label_file.replace('LABEL/', '').rstrip('.txt'), line.split(' ')[0])
                    label = label_map(line)
                    concise_labels.append({'index': index, 'label': label})
        np.save(save_dir + '/{}_{}.npy'.format(cls, 'concise_labels'), concise_labels)


def merge_pred_true(model='retina'):
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        pred_path = 'deployment/{}/{}/deploy_results.npy'.format(model, cls)
        true_path = 'data/Deploy/LABEL/{}_concise_labels.npy'.format(cls)
        pred, true = [np.load(p) for p in [pred_path, true_path]]
        merge = []
        for p, t in tqdm(zip(pred, true)):
            m = {**p, **t}
            merge.append(m)
        np.save(pred_path.replace('deploy_', 'merged_'), merge)


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
    merge_pred_true()
