import os
import cv2
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from operator import itemgetter
from helper.deploy import check_dir
from joblib import Parallel, delayed


def draw_boxes(record, im_save_dir, judge=True):
    """Drawing bounding boxes from a dictionary record
    Argument:
        record: (dict)  E.g., {
         'index': 'data/Deploy/KLFE/KLFE0001/KLFE0001_7.jpg',
         'prediction': [{'class': 'sp_gg', 'score': 0.9886283278465271, 'position': ((456, 230), 477, 199)}],
         'label': 'sp_gg'}
    """
    image = cv2.imread(record['index'])
    h, w = image.shape[:2]
    fig = plt.figure(figsize=(w / 96, h / 96))
    ax = fig.add_subplot(1, 1, 1)

    prediction = record['prediction'][0]
    coords = prediction['position']  # ((x, y), width, height)
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='cyan', linewidth=2))

    display_coords = np.array(coords[0]) + [3, -10]  # align rectangle box, top-left
    display_txt = '{!s}: {:.3f}'.format(prediction['class'], prediction['score'])
    ax.text(*display_coords, display_txt, bbox={'facecolor': 'cyan', 'alpha': 0.4})

    if judge:
        judge_coords = display_coords + [coords[1] - 20, 0]  # align rectangle box, top-right
        is_right = int(prediction['class'] == record['label'])
        judge_txt = [{'symbol': '×', 'color': 'red'}, {'symbol': '√', 'color': 'green'}][is_right]
        ax.text(*judge_coords, judge_txt['symbol'], bbox={'facecolor': judge_txt['color'], 'alpha': 0.7})

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.axis('off')
    plt.imshow(image)
    im_save_name = im_save_dir + record['index'].split('/')[-1]
    plt.savefig(im_save_name, bbox_inches=extent)
    plt.clf()


def draw_boxes_by_model(model, vis_mode='best'):
    """Drawing bouding boxes by model and type
    Argument:
        model: (str) E.g., 'retina/FPN101', 'ssd300'
        vis_mode: (str) current support mode in ['best', 'merged', 'interest']
    """
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        base_path = 'deployment/{}/{}/'.format(model, cls)
        res_path = base_path + '{}_results.npy'.format(vis_mode)
        try:
            results = np.load(res_path)
        except FileNotFoundError:
            print('Do not exist {} predicted file, skipping \'{}\'...'.format(vis_mode, res_path))
            continue

        im_save_dir = base_path + '{}_images/'.format(vis_mode)
        check_dir(im_save_dir)
        # for record in tqdm(results):
        #     draw_boxes(record, im_save_dir, judge=True)
        Parallel(n_jobs=24)(delayed(draw_boxes)(record, im_save_dir) for record in tqdm(results))


if __name__ == '__main__':
    exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
    os.chdir(exp_root)
    draw_boxes_by_model('retina/FPN50', vis_mode='best')
