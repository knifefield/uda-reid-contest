from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from mmt import datasets
from mmt import models
from mmt.eval_reid import eval_func
import torchvision.transforms as T
from mmt.utils.data.dataset_loader import ImageDataset
from mmt.utils.logging import Logger
from mmt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, im_path = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def get_data(name, data_dir, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transforms = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    dataset = datasets.create(name, data_dir)

    query_set = ImageDataset(dataset.query, test_transforms)
    gallery_set = ImageDataset(dataset.gallery, test_transforms)

    query_loader = DataLoader(
        query_set, batch_size=batch_size, shuffle=False,
        collate_fn=val_collate_fn, num_workers=workers, pin_memory=True
    )
    gallery_loader = DataLoader(
        gallery_set, batch_size=batch_size, shuffle=False,
        collate_fn=val_collate_fn, num_workers=workers, pin_memory=True
    )

    return query_loader, gallery_loader


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def test(model, test_loader):
    model.eval()
    feature = []
    cams = []
    pids = []
    with torch.no_grad():
        for data, target, cam in test_loader:
            n, c, h, w = data.size()
            ff = torch.FloatTensor(n, 2048).zero_().to('cuda')
            for i in range(2):
                if i == 1:
                    data = fliplr(data)
                img = data.to('cuda')
                outputs = model(img)
                f = outputs
                ff = ff + f
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            feature.append(ff)
            cams.extend(cam)
            pids.extend(target)

    feature = torch.cat(feature, dim=0).cpu().numpy()
    pids = np.array(pids)
    cams = np.array(cams)
    return feature, pids, cams


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    query_loader, gallery_loader = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model)
    start_epoch = checkpoint['epoch']
    best_mAP = checkpoint['best_mAP']
    print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    q_f, q_id, q_cam = test(model, query_loader)
    g_f, g_id, g_cam = test(model, gallery_loader)

    q_g_dist = np.dot(q_f, np.transpose(g_f))
    q_g_dist = 2. - 2 * q_g_dist  # change the cosine similarity metric to euclidean similarity metric
    all_cmc, mAP = eval_func(q_g_dist, q_id, g_id, q_cam, g_cam)
    all_cmc = all_cmc * 100
    print('rank-1: {:.4f} rank-5: {:.4f} rank-10: {:.4f} rank-20: {:.4f} rank-50: {:.4f} mAP: {:.4f}'.format(all_cmc[0],
                                                                                                             all_cmc[4],
                                                                                                             all_cmc[9],
                                                                                                             all_cmc[
                                                                                                                 19],
                                                                                                             all_cmc[
                                                                                                                 49],
                                                                                                             mAP * 100))

    indices = np.argsort(q_g_dist, axis=1)
    np.savetxt("answer.txt", indices[:, :100], fmt="%04d")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, required=True,
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    main()
