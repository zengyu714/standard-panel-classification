import os

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

from retina.encoder import DataEncoder
from retina.retinanet import RetinaNet
from helper.config import VOC_CLASSES as labels
from helper.deploy import check_dir, DeployDataset, merge_pred_true, statistics

# Init
# -------------------------
exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
os.chdir(exp_root)

base = 'FPN50'
checkpoint = torch.load('retina/checkpoints/{}/best_ckpt.pth'.format(base))

print('Loading model from epoch {}...'.format(checkpoint['epoch']))
net = RetinaNet()
net.load_state_dict(checkpoint['net'])
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.eval().cuda()


def save_deployment(dataset, save_dir, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    total_num = len(dataset)
    h, w = dataset.input_size
    ori_h, ori_w = dataset.image_size

    encoder = DataEncoder()

    deploy_results = []
    for im_batch, p_batch in tqdm.tqdm(dataloader, total=total_num // batch_size, unit=' batch({})'.format(batch_size)):
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
    np.save(save_dir + '/{}.npy'.format('deploy_results'), deploy_results)


def main():
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        save_dir = 'deployment/retina/{}/{}'.format(base, cls)
        check_dir(save_dir)

        # dataset = DeployDataset(cls=cls)
        # save_deployment(dataset, save_dir)
        # merge_pred_true('retina', cls)
        statistics('retina', cls)
        print('*' * 30, 'Finished process {}'.format(cls), '*' * 30)


if __name__ == '__main__':
    main()
