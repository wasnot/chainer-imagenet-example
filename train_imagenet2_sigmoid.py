#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import random

import numpy as np
import pandas as pd

import chainer
from chainer import training
from chainer.training import extensions

import googlenet_sigmoid
import googlenet
import resnet50_2_sigmoid
import resnet50_2


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        pairs = []
        for i, row in pd.read_csv(path).iterrows():
            pairs.append((row.path, row[2:].values))
        
        self.base = chainer.datasets.LabeledImageDataset(pairs, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label
    

def copy_model(src, dst):
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, chainer.link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.link.Link):
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0] or a[1].shape != b[1].shape:
                    print('Ignore {} because of parameter mismatch: {} != {}'.format(child.name, (a[0],a[1].shape), (b[0],b[1].shape)))
                    continue
                b[1].data = a[1].data


def main():
    archs = {
        'googlenet': googlenet_sigmoid.GoogLeNet,
        'resnet50': resnet50_2_sigmoid.ResNet50
    }
    archs_softmax = {
        'googlenet': googlenet.GoogLeNet,
        'resnet50': resnet50_2.ResNet50
    }
    
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='googlenet',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--optimizer', default='adam',
                        help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--lr_shift', type=float, default=0.5,
                        help='lr exponential shift. 0 mean not to shift')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    
    model_cls = archs[args.arch]
    model_softmax_cls = archs_softmax[args.arch]

    # Load the datasets and mean file
    insize = model_cls.insize
    mean = np.load(args.mean)
    train = PreprocessedDataset(args.train, args.root, mean, insize)
    val = PreprocessedDataset(args.val, args.root, mean, insize, False)

    # Initialize the model to train
    model = model_cls(output_size=len(train.get_example(0)[1]))
    if args.initmodel:
        print('Load model from', args.initmodel)
        try:
            chainer.serializers.load_npz(args.initmodel, model)
        except ValueError as e:
            print('not match model. try default ModelClass. "{}"'.format(e))
            src_model = model_softmax_cls()
            chainer.serializers.load_npz(args.initmodel, src_model)
            copy_model(src_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()
    
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    # optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    if args.optimizer == 'adam':
        optimizer = chainer.optimizers.Adam()
    else:
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1 if args.test else 100000), 'iteration'
    log_interval = (1 if args.test else 1000), 'iteration'
    test_interval = 1, 'epoch'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=test_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    #trainer.extend(extensions.snapshot(), trigger=val_interval)
    #trainer.extend(extensions.snapshot_object(
    #    model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model, filename='model_epoch-{.updater.epoch}'))
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=test_interval))
    trainer.extend(extensions.observe_lr(), trigger=test_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=test_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    
    if args.lr_shift > 0:
        # Reduce the learning rate by half every 25 epochs.
        if args.optimizer == 'adam':
            trainer.extend(extensions.ExponentialShift('alpha', args.lr_shift), trigger=(25, 'epoch'))
        else:
            trainer.extend(extensions.ExponentialShift('lr', args.lr_shift), trigger=(25, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
