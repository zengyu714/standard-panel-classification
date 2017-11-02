import os
import sys
import glob
import codecs

import cv2
import tqdm
import visdom
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from joblib import Parallel, delayed

from ssd300.ssd import build_ssd
from helper.config import VOC_CLASSES as labels
from helper.deploy import DeployDataset, check_dir

# Init
# -------------------------
exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
os.chdir(exp_root)

torch.cuda.set_device(1)

# Build SSD300 in test phase
model_resume_step = 135000
num_classes = len(labels) + 1
net = build_ssd('test', 300, num_classes).eval().cuda()
net.load_weights('ssd300/weights/ssd300_0712_{}.pth'.format(model_resume_step))


# Helper function
# -----------------------------------------------------------------------------------------------
def split_frame_index(p):
    """Get frame index from full image path
    E.g.,
    'data/Deploy/KLAC/KLAC0570/KLAC0570_107.jpg' ==> 107
    """
    return int(p.split('_')[-1][:-4])


def split_video_name(p):
    """Get video name from full image path
    E.g.,
    'data/Deploy/KLAC/KLAC0570/KLAC0570_107.jpg' ==> 'KLAC0570'
    """
    return p.split('/')[-2]


def label_map(line):
    """Convert label in Chinese into digital representation
    if label_verbose == 'KLAC0204_4.jpg 上腹部横切面非标准' and class == 'sfb'

    ** Notice: Only cares about three specific classes: 上腹部, 股骨, 丘脑
    """
    if '非' in line:
        return 'negative'
    elif '标准' in line:
        if '上腹部' in line:
            return 'sfb'
        elif '股骨' in line:
            return 'gg'
        elif '丘脑' in line:
            return 'qn'
        # elif '侧脑室' in line:
        #     return 'cns'
        # elif '胆囊' in line:
        #     return 'dn'
        # elif '小脑' in line:
        #     return 'xn'
        else:
            return 'irrelevant'
    else:
        return 'others'


def abs_label_map(line):
    """Convert label in Chinese into label's representation
    if label_verbose == 'KLAC0204_4.jpg 上腹部横切面非标准' and class == 'nsp_sfb'

    ** Notice: Consider all labels for later process
    """
    # Determine the prefix
    s = 'sp_'
    if '非' in line:
        s = 'nsp_'
    # Concat the specific class
    if '标准' in line:
        if '上腹部' in line:
            s += 'sfb'
        elif '股骨' in line:
            s += 'gg'
        elif '丘脑' in line:
            s += 'qn'
        elif '侧脑室' in line:
            s += 'cns'
        elif '胆囊' in line:
            s += 'dn'
        elif '小脑' in line:
            s += 'xn'
    else:
        s = 'others'
    return s


def info_map(record):
    """Convert info in record into summary representation
    + 'all negative labels' to
      '==> ==> ==> No standard panel exists'
    + 'failure' to
      '==> ==> ==> Oops... Fail to find standard panel'
    """
    if record == 'all negative labels':
        return '==> ==> ==> No standard panel exists'
    elif record == 'failure':
        return '==> ==> ==> Oops... Fail to find standard panel'


def display_verbose(verbose):
    if verbose:  # identity
        return lambda a: a
    else:
        return tqdm.tqdm


def inner(nested_dict):
    """
    A typical standard record is a nested dictionary looks like:
    {'index': 'data/Deploy/KLHC/KLHC0002/KLHC0002_12.jpg',
     'prediction': [{'class': 'sp_sfb',
                     'position': ((467.21396, 164.32651),
                                   449.51553344726562,
                                   438.98019409179688),
                     'score': 0.9994955062866211}
                    ]}
    """
    return nested_dict['prediction'][0]


def get_highest_score(records):
    """Find the highest score from a list of frame predication in the same class
    Only consider the valid class: 'sp_qn', 'sp_gg', 'sp_sfb'
    """
    valid = ['sp_qn', 'sp_gg', 'sp_sfb']
    records = [r for r in records if inner(r)['class'] in valid]
    return sorted(records, key=lambda d: inner(d)['score'])[-1]


def choose_from_multiboxes(full_predictions):
    """Choose the highest score from multiple boundingboxes in one frame"""
    predictions = full_predictions['prediction']
    return sorted(predictions, key=lambda k: k['score'])[-1]


def draw_boundingbox(image, prediction, im_save_name, show_label=None):
    h, w = image.shape[:2]
    fig = plt.figure(figsize=(w / 96, h / 96))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')

    plt.imshow(image)

    display_txt = '{!s}: {:.5f}'.format(prediction['class'], prediction['score'])
    if show_label is not None:
        display_txt += ' v.s. {}'.format(show_label)
    coords = prediction['position']  # ((x, y), width, height)
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='cyan', linewidth=2))
    ax.text(*coords[0], display_txt, bbox={'facecolor': 'cyan', 'alpha': 0.4})

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(im_save_name, bbox_inches=extent)
    plt.clf()


# -----------------------------------------------------------------------------------------------


def save_deployment(dataset, save_dir, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    total_num = len(dataset)
    h, w = dataset.image_size

    deploy_results = []
    for im_batch, p_batch in tqdm.tqdm(dataloader,
                                       total=total_num // batch_size, unit=' batch({})'.format(batch_size)):
        x_batch = Variable(im_batch, volatile=True).cuda()

        detections = net(x_batch).data
        scale = torch.Tensor([w, h, w, h])  # scale each detection back up to the image

        for i, detection in enumerate(detections):
            pred = []
            for j in range(detection.size(0)):
                k = 0
                while detection[j, k, 0] >= 0.6:
                    label_name = labels[j - 1]
                    pt = (detection[j, k, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    k += 1
                    # score = detection[j, k, 0]
                    # pred.append({'class': label_name, 'score': score, 'position': coords})
                    pred.append({'class': label_name, 'position': coords})
            deploy_results.append({'index': p_batch[i], 'prediction': pred})
    np.save(save_dir + '/{}.npy'.format('deploy_results'), deploy_results)


# Measure 1: strictly compare with label
# -----------------------------------------------------------------------------------------------

def compare_with_label(cls='KLHC', verbose=False):
    deploy_file = 'ssd300/eval/kl/{}/deploy_results.npy'
    deploy_all = np.load(deploy_file.format(cls))
    video_names = sorted([d for d in os.listdir('data/Deploy/' + cls) if not d.endswith('.wmv')])

    for vn in display_verbose(verbose)(video_names):
        deploy_video = [d for d in deploy_all if d['index'].split('/')[-2] == vn]

        label_file = 'data/Deploy/LABEL/{}/{}.txt'.format(cls, vn)
        if not os.path.isfile(label_file):
            continue
        with codecs.open(label_file, encoding='gbk') as f:
            lines = f.readlines()

        compare_results = []
        persist_results = []
        exist_standard = False

        for j, frame_pred in enumerate(deploy_video):
            frame_name = frame_pred['index'].split('/')[-1]
            try:
                label_verbose = lines[j]
                if not label_verbose.startswith(frame_name):
                    raise IndexError
            except IndexError:  # label is missing or not corresponding
                print('\n==> ==> ==> Label: {}is NOT compatible with prediction: {}'.format(lines[-1], frame_name))
                break

            if '非' not in label_verbose and '标准' in label_verbose:
                # interest = ['上腹部', '股骨', '丘脑']
                # specific = [i for i in interest if i in label_verbose]
                # if len(specific):
                if '上腹部' in label_verbose or '股骨' in label_verbose or '丘脑' in label_verbose:
                    exist_standard = True  # see if all labels are negative

            dicts = frame_pred['prediction']  # should be a list
            if len(dicts) > 1:
                dicts = sorted(dicts, key=lambda d: d['score'])[-1]  # leave the highest score
            elif len(dicts) == 1:
                dicts = dicts[0]
            elif len(dicts) == 0:
                continue

            # Consider doctor's labels
            pred_temp = dicts['class']
            label_temp = label_map(label_verbose)
            correct = pred_temp.startswith('sp') and pred_temp.endswith(label_temp)
            # Despite doctor's labels
            despite = pred_temp in ['sp_qn', 'sp_gg', 'sp_sfb']

            if correct:  # once find a standard panel, it means success
                compare_results.append(frame_pred)
            elif despite:
                persist_results.append(frame_pred)

        # Save results
        compare_save_dir = 'ssd300/eval/kl/{}/measure_1/compare'.format(cls)
        check_dir(compare_save_dir)
        compare_filename = compare_save_dir + '/{}.npy'.format(vn)

        info = ''
        if not exist_standard:
            compare_results.append('all negative labels')
            info = '<Strict> No standard panel exists in video: {}'.format(vn)
        elif len(compare_results) == 0:
            compare_results.append('failure')
            info = '<Strict> Fail to detect standard panel in video: {}'.format(vn)

        # Other than 'all negative labels' or 'failure'
        if len(compare_results) == 1 and len(persist_results):
            compare_results.extend(persist_results)
            info += '\n******** Suppose errors in labels...'

        if len(info) and verbose:
            print(info)
        np.save(compare_filename, compare_results)


def find_standard_panel(cls='KLHC', draw_bbox=True, verbose=False):
    with open('ssd300/eval/kl/{}/measure_1/{}_summary.txt'.format(cls, cls), 'w') as f:
        record_paths = glob.glob('ssd300/eval/kl/{}/compare/*.npy'.format(cls))
        nums = len(record_paths)
        for i, record_path in display_verbose(verbose)(enumerate(sorted(record_paths), start=1)):
            record = np.load(record_path)

            is_write_path = False  # also means to challenge doctor's label
            if record[0] in ['all negative labels', 'failure']:
                info = info_map(record[0])
                if verbose:
                    print('{} in {}  {}/{}'.format(info, record_path, i, nums))
                f.write('{}: {} '.format(record_path, info))
                is_write_path = True
                record = np.delete(record, 0)
            # both neg in labels and prediction
            if len(record) == 0:
                f.write('\r\n')
                continue
            # compute highest score
            std = get_highest_score(record)
            if not is_write_path:
                f.write('{}: {}\r\n'.format(record_path, std))
            else:
                f.write('<Suppose>: {}\r\n'.format(std))

            # Plot image along with boundingbox
            if draw_bbox:
                vn = record_path.split('/')[-1][:-4]  # video_name
                prediction = inner(std)
                image = cv2.imread(std['index'])

                if not is_write_path:
                    im_save_dir = 'ssd300/eval/kl/{}/measure_1/same_images'.format(cls)
                else:
                    im_save_dir = 'ssd300/eval/kl/{}/measure_1/diff_images'.format(cls)
                check_dir(im_save_dir)
                im_save_name = '{}/{}_{}.jpg'.format(im_save_dir, vn, split_frame_index(std['index']))

                draw_boundingbox(image, prediction, im_save_name)


def final_statistics():
    with open('ssd300/eval/kl/statistics.txt', 'w') as f_stat:
        for cls in ['KLAC', 'KLFE', 'KLHC']:
            with open('ssd300/eval/kl/{}/{}_summary.txt'.format(cls, cls)) as f:
                lines = f.readlines()

            real_num = len(lines) // 2
            lines = lines[real_num:]

            lines = [l for l in lines if 'No' not in l]
            acc = 1 - len([l for l in lines if 'Oops' in l]) / len(lines)
            info = '{}({}) accuracy: {:.5f}\r\n'.format(cls, real_num, acc)
            f_stat.write(info)
            print(info)


def deploy_1():
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        save_dir = 'deployment/ssd300/{}'.format(cls)
        check_dir(save_dir)
        dataset = DeployDataset(cls=cls)
        save_deployment(dataset, save_dir)
        # compare_with_label(cls)
        # find_standard_panel(cls, draw_bbox=True)
        print('*' * 30, 'Finished process {}'.format(cls), '*' * 30)
        # final_statistics()


# -----------------------------------------------------------------------------------------------


# Measure 2
# -----------------------------------------------------------------------------------------------
def compare_per_frame(cls='KLHC'):
    deploy_file = 'ssd300/eval/kl/{}/deploy_results.npy'
    deploy_all = np.load(deploy_file.format(cls))
    video_names = sorted([d for d in os.listdir('data/Deploy/' + cls) if not d.endswith('.wmv')])

    Parallel(n_jobs=24)(delayed(do_save_bbox)(vn, deploy_all, cls) for vn in tqdm.tqdm(video_names))


def do_save_bbox(vn, deploy_all, cls):
    deploy_video = [d for d in deploy_all if d['index'].split('/')[-2] == vn]
    label_file = 'data/Deploy/LABEL/{}/{}.txt'.format(cls, vn)
    if os.path.isfile(label_file):  # continue when label file exists
        with codecs.open(label_file, encoding='gbk') as f:
            lines = f.readlines()

        dual_results = []
        for j, frame_pred in enumerate(deploy_video):
            try:
                prediction = choose_from_multiboxes(frame_pred)
            except IndexError:
                continue  # blank prediction

            frame_path = frame_pred['index']
            frame_name = frame_path.split('/')[-1]
            try:
                label_verbose = lines[j]
                if not label_verbose.startswith(frame_name):
                    raise IndexError
            except IndexError:  # label is missing or not corresponding
                print('\n==> ==> ==> Label: {}is NOT compatible with prediction: {}'.format(lines[-1], frame_name))
                break

            true_temp = abs_label_map(label_verbose)
            pred_temp = prediction['class']

            dual_results.append({'index': frame_name, 'true': true_temp, 'pred': pred_temp})

            im_save_dir = 'ssd300/eval/kl/{}/measure_2/images/{}'.format(cls, vn)
            check_dir(im_save_dir)
            im_save_name = '{}/{}'.format(im_save_dir, frame_name)

            image = cv2.imread(frame_path)
            draw_boundingbox(image, prediction, im_save_name, show_label=true_temp)

        dual_save_name = 'ssd300/eval/kl/{}/measure_2/dual_compare_results/{}.npy'.format(cls, vn)
        np.save(dual_save_name, dual_results)


def deploy_2():
    # for cls in ['KLAC', 'KLFE', 'KLHC']:
    for cls in ['KLHC']:
        measure_dir = 'ssd300/eval/kl/{}/measure_2'.format(cls)
        check_dir(measure_dir)
        check_dir(measure_dir + '/images')
        check_dir(measure_dir + '/dual_compare_results')

        # dataset = DeployDataset(cls=cls)
        # save_deployment(dataset)
        compare_per_frame(cls)
        print('*' * 30, 'Finished process {}'.format(cls), '*' * 30)


if __name__ == '__main__':
    deploy_1()
