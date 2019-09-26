from __future__ import print_function

import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from ssd import build_ssd
from data import VOC_ROOT, COCO_ROOT, BaseTransform
from data import COCOAnnotationTransform, COCODetection, COCO_CLASSES
from data import VOCAnnotationTransform, VOCDetection, VOC_CLASSES
from data import AmanoDetection


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--type', default='VOC', choices=['VOC', 'COCO', 'AMANO'],
                    type=str, help='VOC, COCO and AMANO')
parser.add_argument('--dataset', default=VOC_ROOT,
                    help='Dataset root directory path')

parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--dest', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_net(dest, net, cuda, testset, transform, thresh: float = .6):
    for index in tqdm(range(len(testset))):
        img = testset.pull_image(index)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        # get first of batch (in case of test set, batch size always 1)
        detections = net(x).data[0]

        outputs = np.empty((0, 6), dtype=np.float32)

        for klass, boxes in enumerate(detections):
            candidates = boxes[boxes[:, 0] >= thresh]

            if candidates.size(0) == 0:
                continue

            candidates[:, 1:] *= scale

            outputs = np.concatenate((
                outputs,
                np.hstack((
                    np.full((np.size(candidates, 0), 1), klass, dtype=np.uint8),
                    candidates.cpu().numpy(),
                )),
            ))

        pd.DataFrame(outputs).to_csv(dest.joinpath(f'{index:06d}.txt'), header=None)


def test_voc():
    # load net
    if args.type == 'COCO':
        testset = COCODetection(root=args.dataset)
    elif args.type == 'VOC':
        testset = VOCDetection(root=args.dataset)
    elif args.type == 'AMANO':
        testset = AmanoDetection(root=args.dataset)

    num_classes = testset.cfg['num_classes']
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    net.eval()
    print('Finished loading model!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    test_net(dest, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()
