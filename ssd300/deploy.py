import os
import tqdm
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


from ssd300.ssd import build_ssd
from helper.config import VOC_CLASSES as labels
from helper.deploy import DeployDataset, check_dir, merge_pred_true, statistics

# Init
# -------------------------
exp_root = '/home/zengyu/Lab/pytorch/standard-panel-classification'
os.chdir(exp_root)

torch.cuda.set_device(2)

# Build SSD300 in test phase
model_resume_step = 120000
num_classes = len(labels) + 1
net = build_ssd('test', 300, num_classes).eval().cuda()
net.load_weights('ssd300/weights/ssd300_0712_{}.pth'.format(model_resume_step))


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
                    score = detection[j, k, 0]
                    pt = (detection[j, k, 1:] * scale).cpu().numpy().astype(int)
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    k += 1
                    pred.append({'class': label_name, 'score': score, 'position': coords})
            deploy_results.append({'index': p_batch[i], 'prediction': pred})
    np.save(save_dir + '/{}.npy'.format('deploy_results'), deploy_results)


def main():
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        save_dir = 'deployment/ssd300/{}'.format(cls)
        check_dir(save_dir)

        # dataset = DeployDataset(cls=cls)
        # save_deployment(dataset, save_dir)
        # merge_pred_true('ssd300', cls)
        statistics('ssd300', cls)
        print('*' * 30, 'Finished process {}'.format(cls), '*' * 30)


if __name__ == '__main__':
    main()
