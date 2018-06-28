import cv2
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import dataset, Chain, training, optimizers, \
    iterators, reporter, cuda, serializers
import argparse
import glob
import copy
import random
import os

if cuda.available:
    xp = cuda.cupy
else:
    xp = np


class VGG(Chain):
    def __init__(self):
        super(VGG, self).__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.upsample3 = L.Deconvolution2D(
                256, 2, ksize=1, stride=1, pad=0)
            self.upsample4 = L.Deconvolution2D(
                512, 2, ksize=4, stride=2, pad=1)
            self.upsample5 = L.Deconvolution2D(
                512, 2, ksize=8, stride=4, pad=2)
            self.upsample = L.Deconvolution2D(
                2, 1, ksize=16, stride=8, pad=4)

    def __call__(self, x):
        h1 = self.base(x, layers=['pool3'])['pool3']
        h2 = self.base(x, layers=['pool4'])['pool4']
        h3 = self.base(x, layers=['pool5'])['pool5']
        h1 = self.upsample3(h1)
        h2 = self.upsample4(h2)
        h3 = self.upsample5(h3)
        return self.upsample(h1 + h2 + h3)


class DataSet(dataset.DatasetMixin):
    def __init__(self, size, path0, path1):
        self.data = []
        self.size = size
        IMG_PATHS = [path0, path1]
        self.base_n = len(glob.glob(IMG_PATHS[1]))
        self.img_n = len(glob.glob(IMG_PATHS[0]))

        for i, IMG_PATH in enumerate(IMG_PATHS):
            self.data.append([])
            for path in glob.glob(IMG_PATH):
                img_ = cv2.imread(path)
                if i == 0:
                    img = cv2.resize(img_, (256, 256))
                else:
                    img = cv2.resize(img_, (512, 512))
                self.data[i].append(img)

    def __len__(self):
        return self.size

    def get_example(self, i):
        base = copy.deepcopy(self.data[1][i % self.base_n])
        offset_x = random.randint(0, 255)
        offset_y = random.randint(0, 255)

        base[offset_x:offset_x + 256, offset_y:offset_y + 256, :] =\
            copy.deepcopy(self.data[0][i % self.img_n])
        mask = np.zeros((512, 512))
        mask[offset_x:offset_x + 256, offset_y:offset_y + 256] =\
            np.ones((256, 256))
        return xp.array(cv2.resize(base, (224, 224)).
                        transpose(2, 0, 1) / 255).astype('float32'),\
            xp.array([cv2.resize(mask, (224, 224))]).astype('float32')


class Loss_Link(Chain):
    def __init__(self, model):
        super(Loss_Link, self).__init__()
        self.y = None
        with self.init_scope():
            self.predictor = model

    def __call__(self, x, t):
        self.y = self.predictor(x)
        self.mean_loss = F.mean_squared_error(self.y, t)
        reporter.report({'mean_loss': self.mean_loss}, self)
        self.worst_loss = F.max(F.squared_error(self.y, t))
        reporter.report({'worst_loss': self.worst_loss}, self)
        return self.mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of examples in epoch')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--model', '-m', default='',
                        help='Load model')
    parser.add_argument('--path0', '-p0', default='./DATA/*/*.png',
                        help='path for images used subject')
    parser.add_argument('--path1', '-p1', default='./DATA/*/*.JPEG',
                        help='path for images used back')
    parser.add_argument('--test', '-t', action='store_true',
                        help='evaluation only')
    parser.add_argument('--image', action='store_true',
                        help='put image for test')

    args = parser.parse_args()

    train_dataset = DataSet(420, args.path0, args.path1)
    test_dataset = DataSet(50, args.path0, args.path1)

    model = Loss_Link(VGG())

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    model.predictor.base.disable_update()

    if args.model:
        serializers.load_npz(args.model, model)

    train_iter = iterators.SerialIterator(
        train_dataset, batch_size=args.batchsize, shuffle=True)
    test_iter = iterators.SerialIterator(
        test_dataset,  batch_size=args.batchsize, repeat=False)

    if args.test:
        eva = training.extensions.Evaluator(
            test_iter, model, device=args.gpu)()
        for key in eva:
            print(key + ":" + str(eva[key]))
    elif args.image:
        if not os.path.exists(args.out):
            os.mkdir(args.out)
        IMG_PATHS = [args.path0, args.path1]
        data = []
        base_n = len(glob.glob(IMG_PATHS[1]))
        img_n = len(glob.glob(IMG_PATHS[0]))
        for i, IMG_PATH in enumerate(IMG_PATHS):
            data.append([])
            for path in glob.glob(IMG_PATH):
                img_ = cv2.imread(path)
                if i == 0:
                    img = cv2.resize(img_, (256, 256))
                else:
                    img = cv2.resize(img_, (512, 512))
                data[i].append(img)

        offset_x = random.randint(0, 255)
        offset_y = random.randint(0, 255)
        base = copy.deepcopy(data[1][random.randint(0, base_n - 1)])
        base[offset_x:offset_x + 256, offset_y:offset_y +
             256, :] = copy.deepcopy(data[0][random.randint(0, img_n - 1)])

        cv2.imwrite(args.out + "/input_image.png",
                    cv2.resize(base, (224, 224)))
        mask = np.zeros((512, 512))
        mask[offset_x:offset_x + 256, offset_y:offset_y + 256] =\
            np.ones((256, 256))
        cv2.imwrite(args.out + "/ideal_image.png",
                    cv2.resize(np.array(mask * 255).
                               astype("uint8"), (224, 224)))
        pred = model.predictor(xp.array([cv2.resize(base, (224, 224)).
                                         transpose(2, 0, 1) / 255]).
                               astype('float32')).array[0] > 0.7
        cv2.imwrite(args.out + "/output_image.png",
                    np.array(pred * 255).reshape(224, 224).astype("uint8"))
    else:
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(
            updater, (args.epoch, 'epoch'), out=args.out)

        trainer.extend(training.extensions.Evaluator(
            test_iter, model, device=args.gpu),
            trigger=(1, 'epoch'))

        trainer.extend(training.extensions.LogReport(
            trigger=(1, 'epoch')))
        trainer.extend(training.extensions.PrintReport(
            entries=['iteration', 'main/loss',
                     'main/accuracy', 'elapsed_time']),
            trigger=(1, 'epoch'))
        trainer.extend(training.extensions.snapshot(),
                       trigger=((1, 'epoch')))
        if args.resume:
            serializers.load_npz(args.resume, trainer, strict=False)
        trainer.run()
        serializers.save_npz('model.npz', model)


if __name__ == "__main__":
    main()
