import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from tqdm import tqdm
from operator import itemgetter
from joblib import Parallel, delayed
from sklearn.manifold import TSNE

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
os.chdir(exp_root)

from retina.fpn import FPN50
from helper.deploy import check_dir
from helper.ultrasound_ops import BaseTransform
from helper.input import AnnotationTransform, SSDDataset, ssd_collate


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


def draw_boxes_by_model(model, vis_name='corrected_best'):
    """Drawing bouding boxes by model and type
    Argument:
        model: (str) E.g., 'retina/FPN101', 'ssd300'
        vis_mode: (str) current support mode in ['best', 'merged', 'interest']
    """
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        base_path = 'deployment/{}/{}/'.format(model, cls)
        res_path = base_path + '{}.npy'.format(vis_name)
        try:
            results = np.load(res_path)
        except FileNotFoundError:
            print('Do not exist {} predicted file, skipping \'{}\'...'.format(vis_name, res_path))
            continue

        im_save_dir = base_path + '{}_images/'.format(vis_name)
        check_dir(im_save_dir)
        # for record in tqdm(results):
        #     draw_boxes(record, im_save_dir, judge=True)
        Parallel(n_jobs=24)(delayed(draw_boxes)(record, im_save_dir) for record in tqdm(results))


# Prepare data for visualization
def get_loss(loss_file, mode='test'):
    """Find loss value in the log file of console outputs.
    Argument:
        loss_file: (str) path to log loss file
    Return:
        list of loss
    """
    with open(loss_file, 'r+') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]  # remove new line character
    lines = [l for l in lines if l and not l.startswith('===>')][1:]  # remove blank lines

    train_end_idx = np.array([idx - 1 for idx, l in enumerate(lines) if l.startswith('Test')])
    test_end_idx = np.array([idx - 2 for idx, l in enumerate(lines) if l.startswith('Epoch')])[1:]

    # Not stable if use beginning and middle data
    # --------------------------------------------------
    # >>>     if mode == 'train':
    # >>>         end_idx = train_end_idx
    # >>>         epoch_nums = end_idx[0] - end_idx[0] - 2
    # >>>     elif mode == 'test':
    # >>>         end_idx = test_begin_idx
    # >>>         epoch_nums = end_idx[1] - end_idx[0] - 2
    # >>>         print(epoch_nums)
    #
    # >>>     mid_idx = end_idx + epoch_nums // 2
    # >>>     idx = []
    # >>>     for tp in list(zip(end_idx, mid_idx)):
    # >>>         idx += tp
    # >>>     idx = idx[:-1]
    #
    # >>>     loss_line = itemgetter(*idx)(lines)

    loss_line = itemgetter(*eval(mode + '_end_idx'))(lines)
    loss = [float(l.split(':')[-1].strip()) for l in loss_line]
    return loss


def load_array(epoch, batch_size=16):
    """Load tensor data from dataloader
    Argument:
        epoch: (int) load how many epochs
        batch_size: (int)
    Return:
        images: list of numpy.array, length == epoch * batch_size
        labels: list of numpy.array, length == epoch * batch_size
    """
    means = (104, 117, 123)
    ssd_dim = 300
    train_sets = [('2007', 'trainval')]
    voc_root = '/home/zengyu/Lab/pytorch/standard-panel-classification/data/'

    trainset = SSDDataset(voc_root,
                          train_sets,
                          BaseTransform(ssd_dim, means),
                          AnnotationTransform())
    trainloader = data.DataLoader(trainset,
                                  shuffle=True,
                                  num_workers=4,
                                  batch_size=batch_size,
                                  collate_fn=ssd_collate)
    images = []
    labels = []
    for batch_idx, (list_im, list_lb) in enumerate(trainloader, 1):
        batch_im = np.array([im.numpy() for im in list_im])
        batch_lb = np.array([lb.numpy() for lb in list_lb])

        images.append(batch_im)
        labels.append(batch_lb)
        if batch_idx == epoch:
            break
    return images, labels


def image_tsne(epoch, save_name='helper/image_tsne.npy'):
    images, labels = load_array(epoch)
    print('Finished loading image data...')
    X, Y = [np.concatenate(item) for item in [images, labels]]
    X, Y = [item.reshape((item.shape[0], -1)).astype(int) for item in [X, Y]]
    Y = Y[:, [-1]]  # index the class without losing dim

    tsne = TSNE(n_components=2, perplexity=30, verbose=2).fit_transform(X)
    tsne_Y = np.concatenate([tsne, Y], 1)
    np.save(save_name, tsne_Y)

    # Draw demo
    # --------------------------------------------------------------------------
    # vis_x, vis_y, Y = tsne.T
    # plt.scatter(vis_x, vis_y, c=Y, cmap=plt.cm.get_cmap('jet', 13), alpha=0.7)
    # plt.colorbar(ticks=range(13))
    # plt.clim(-0.5, 12.5)
    # plt.show()


def filtered_tsne(epoch, save_name='helper/filtered_tsne.npy'):
    torch.cuda.set_device(3)

    model = FPN50()
    pretrained_dict = torch.load('retina/checkpoints/FPN50/best_ckpt.pth')['net']
    current_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_dict}
    # 2. overwrite entries in the existing state dict
    current_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(current_dict)

    model = model.cuda().eval()
    print('Finished loading weights...')

    filtered = []
    targets = []
    for _ in tqdm(range(epoch)):
        images, labels = load_array(1)
        # images
        images = [torch.FloatTensor(im) for im in images]  # convert to tensor
        images = Variable(torch.cat(images).cuda())
        out = model(images)[0].data.cpu().numpy()  # choose largest output: [batch_size, 256, 38, 38]
        filtered.append(out.reshape((out.shape[0], -1)))
        # targets
        labels = np.array(labels).squeeze()  # [batch_size, 5]
        targets.append(labels[:, [-1]])

    X, Y = [np.concatenate(item) for item in [filtered, targets]]
    tsne = TSNE(n_components=2, perplexity=30, verbose=2).fit_transform(X)
    tsne_Y = np.concatenate([tsne, Y], 1)    # [image_nums, 3]
    np.save(save_name, tsne_Y)


if __name__ == '__main__':
    draw_boxes_by_model('retina/FPN34', vis_name='corrected_best')

    # image_tsne(300, save_name='helper/image_tsne.npy')
    # filtered_tsne(300, save_name='helper/filtered_tsne.npy')
