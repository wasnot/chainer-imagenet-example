import chainer
import chainer.functions as F
import chainer.links as L


def sigmoid_accuracy(y, t):
    y_tmp = F.split_axis(y, t.shape[1], axis=1)
    acc = []
    for i in range(t.shape[1]):
        acc.append(F.reshape(F.accuracy(F.concat([1. - y_tmp[i].data, y_tmp[i].data]), t[:, i]), (-1, 1)))
    return F.average(F.concat(acc, axis=0))


class GoogLeNet(chainer.Chain):

    insize = 224

    def __init__(self, output_size=1000):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3,  64, 7, stride=2, pad=3)
            self.conv2_reduce = L.Convolution2D(64,  64, 1)
            self.conv2 = L.Convolution2D(64, 192, 3, stride=1, pad=1)
            self.inc3a = L.Inception(192,  64,  96, 128, 16,  32,  32)
            self.inc3b = L.Inception(256, 128, 128, 192, 32,  96,  64)
            self.inc4a = L.Inception(480, 192,  96, 208, 16,  48,  64)
            self.inc4b = L.Inception(512, 160, 112, 224, 24,  64,  64)
            self.inc4c = L.Inception(512, 128, 128, 256, 24,  64,  64)
            self.inc4d = L.Inception(512, 112, 144, 288, 32,  64,  64)
            self.inc4e = L.Inception(528, 256, 160, 320, 32, 128, 128)
            self.inc5a = L.Inception(832, 256, 160, 320, 32, 128, 128)
            self.inc5b = L.Inception(832, 384, 192, 384, 48, 128, 128)
            self.loss3_fc = L.Linear(1024, output_size)

            self.loss1_conv = L.Convolution2D(512, 128, 1)
            self.loss1_fc1 = L.Linear(2048, 1024)
            self.loss1_fc2 = L.Linear(1024, output_size)

            self.loss2_conv = L.Convolution2D(528, 128, 1)
            self.loss2_fc1 = L.Linear(2048, 1024)
            self.loss2_fc2 = L.Linear(1024, output_size)

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)
        loss1 = F.sigmoid_cross_entropy(l, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)
        loss2 = F.sigmoid_cross_entropy(l, t)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4))
        loss3 = F.sigmoid_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        
        accuracy = sigmoid_accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        return loss

    def predict(self, x):
        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            h = F.relu(self.conv1(x))
            h = F.local_response_normalization(
                F.max_pooling_2d(h, 3, stride=2), n=5)
            h = F.relu(self.conv2_reduce(h))
            h = F.relu(self.conv2(h))
            h = F.max_pooling_2d(
                F.local_response_normalization(h, n=5), 3, stride=2)

            h = self.inc3a(h)
            h = self.inc3b(h)
            h = F.max_pooling_2d(h, 3, stride=2)
            h = self.inc4a(h)

            h = self.inc4b(h)
            h = self.inc4c(h)
            h = self.inc4d(h)

            h = self.inc4e(h)
            h = F.max_pooling_2d(h, 3, stride=2)
            h = self.inc5a(h)
            h = self.inc5b(h)

            h = F.average_pooling_2d(h, 7, stride=1)
            h = self.loss3_fc(F.dropout(h, 0.4))
        return F.sigmoid(h)
