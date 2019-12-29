from __future__ import print_function
import os
import time
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import SIXray_CLASSES as labelmap
from data import SIXray_ROOT, SIXrayAnnotationTransform, SIXrayDetection, BaseTransform
from ssd import build_ssd

dataset_mean = (104, 117, 123)
set_type = 'test'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')

parser.add_argument('image_path', type=str, help='Test file')
parser.add_argument('anno_path', type=str, help='Test file')
parser.add_argument('result_folder', type=str, help='File path to save results')

parser.add_argument('--trained_model',
                    default='weights/SIXray.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default='./test_set',
                    help='Location of VOC root directory')
parser.add_argument('--dataset', default=None, type=str,
                    help='Test file')


args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.result_folder):
    os.mkdir(args.result_folder)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(result_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # det_file = os.path.join(result_folder, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w, og_im = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, result_folder, dataset)


def get_voc_results_file_template(filedir, image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset, output_dir):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(output_dir, set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset, output_dir)
    # do_python_eval(output_dir)


def test(img_path, anno_path):
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model,
                                   map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = SIXrayDetection(args.dataset_root, args.dataset,
                              BaseTransform(300, dataset_mean),
                              SIXrayAnnotationTransform(),
                              test_set_path=(img_path, anno_path))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.result_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)


if __name__ == '__main__':
    img_path = args.image_path
    anno_path = args.anno_path
    test(img_path, anno_path)
