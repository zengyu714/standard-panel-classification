import os
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data

from torch.autograd import Variable
from helper.input import RetinaDataset

from retina.loss import FocalLoss
from retina.retinanet import RetinaNet
from bluntools.checkpoints import save_checkpoints

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--model', '-m', default='FPN50', help='base fpn model')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--stop-parallel', '-x', action='store_true', help='stop use parallel training')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')

train_sets = [('2007', 'trainval')]
val_sets = [('2007', 'val')]

trainset = RetinaDataset(train_sets)
testset = RetinaDataset(val_sets)
trainloader = data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8,
                              collate_fn=trainset.collate_fn, pin_memory=True)
testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=8,
                             collate_fn=testset.collate_fn, pin_memory=True)

# Coarse setting
os.chdir('/home/zengyu/Lab/pytorch/standard-panel-classification/')

# Model
base = args.model
net = RetinaNet(base)
net.load_state_dict(torch.load('retina/model/init_{}.pth'.format(base)))

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('retina/checkpoints/{}/best_ckpt.pth'.format(base))
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']

if not args.stop_parallel:
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
else:
    torch.cuda.set_device(3)

net.cuda()

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)

        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('** batch_idx: %d ** | train_loss: %.4f | avg_loss: %.4f\r\n' % (
            batch_idx, loss.data[0], train_loss / (batch_idx + 1)))


# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('** batch_idx: %d ** | test_loss: %.3f | avg_loss: %.3f\r\n' % (
            batch_idx, loss.data[0], test_loss / (batch_idx + 1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        if not args.stop_parallel:
            saved_module = net.module
        else:
            saved_module = net
        state = {
            'net'  : saved_module.state_dict(),
            'loss' : test_loss,
            'epoch': epoch,
        }
        torch.save(state, 'retina/checkpoints/{}/best_ckpt.pth'.format(base))
        best_loss = test_loss


# Check save dir
save_dir = 'retina/checkpoints/{}'.format(base)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    save_checkpoints(net, save_dir, epoch, prefix='retinaNet')
