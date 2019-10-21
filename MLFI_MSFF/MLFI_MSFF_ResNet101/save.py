from __future__ import print_function
# the libraries of System
import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# the libraries for Scientific Computing
import numpy as np
import scipy
import cv2
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd
import torch.backends.cudnn as cudnn

# the libraries for Data Processing
import matplotlib.pyplot as plt

# the libraries for Auxiliary
import argparse
import timeit
import pprint

# my libraries
from networks.MLFI_MSFF_ResNet101 import MLFI_MSFF_ResNet
from utils.loss import CrossEntropy2d, loss_calc
from utils.datasets import ImageSaliencyTestDataSet

###############################################################################
# the next part is the configuration of experiments

"""
# the path of needful files
"""
TEST_DATA_ROOT = '/home/hmk/Datasets/IMAGE/'

TEST_DATASET_LIST = ['DUTS-TE','ECSSD','HKU-IS', 'PASCAL-S','MSRA10K']
TEST_DATASET = TEST_DATASET_LIST[4]
TEST_DATA_DIRECTORY = os.path.join(TEST_DATA_ROOT, TEST_DATASET)

TEST_DATA_LIST = '../dataset/image/' + TEST_DATASET + '.txt'
SNAPSHOT_DIR = './snapshots_MLFI_MSFF_ResNet101/'
OUR_METHOD = '/MLFI_MSFF_ResNet101/'

RESULT_DIRECTORY = '../result/'
SAVE_RESULT_DIRECTORY = os.path.join(RESULT_DIRECTORY, TEST_DATASET) + OUR_METHOD
"""
# some Hyperparameters
"""
NUM_CLASSES = 2
snapPrefix = 'image_saliency_'
ITERATION_NUM = 22 # the iteration num of the best performance model weight
RESTORE_FROM = os.path.join(SNAPSHOT_DIR, snapPrefix+str(ITERATION_NUM)+'.pth')

#IMG_MEAN = np.array((106.28296121819328, 120.42722964272463, 122.48082090431021), dtype=np.float32) # DUTS-TE
#IMG_MEAN = np.array((92.716854889126779, 112.34279191750795, 117.00990821623284), dtype=np.float32) # ECSSD
#IMG_MEAN = np.array((104.14717968961838, 121.61712647247944, 123.50779900571109), dtype=np.float32) # HKU-IS
#IMG_MEAN = np.array((102.48758588318904 , 112.74501913522583, 117.01761990728161), dtype=np.float32) # PASCAL-S
IMG_MEAN = np.array((100.00062249530835, 110.47970065645588, 115.56613997577892), dtype=np.float32) # MSRA10K

###############################################################################
# let's prepare for validating the model!
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=OUR_METHOD[1:-1])
    parser.add_argument("--test-data-dir", type=str, default=TEST_DATA_DIRECTORY,
                        help="Path to the directory containing the DAVIS dataset.")
    parser.add_argument("--test-data-list", type=str, default=TEST_DATA_LIST,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--result-dir", type=str, default=SAVE_RESULT_DIRECTORY,
                        help="Path to the file results.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

###############################################################################
# OK! Let's save the saliency detection results.
    
"""
def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')

def show_all(gt, pred):
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subp#!/usr/bin/env python3
# -*- coding: utf-8 -*-lots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0), 
                    (1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0), 
                    (1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0), 
                    (1.0,1.0,1.0),(1.0,1.0,1.0)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()
"""

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    model = MLFI_MSFF_ResNet(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(args.gpu)
    testloader = data.DataLoader(ImageSaliencyTestDataSet(args.test_data_dir, 
                                                          args.test_data_list, 
                                                          input_size=(321, 321), 
                                                          mean=IMG_MEAN, 
                                                          crop=False, 
                                                          rotate=False,
                                                          validation=False), 
                                    batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    # interp = nn.Upsample(size=(505, 505), mode='bilinear')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, _, size, name = batch
    
            image = image.to(torch.device('cuda', args.gpu))
            # print size# , type(name)
            size = size[0].numpy()
            # print type(size)
            output = model(image)
            
            get_probality_map = nn.Softmax2d()
            output = get_probality_map(output)
            
            output = nn.functional.interpolate(output, size=(size[0], size[1]), mode='bilinear', align_corners=True)
            output = output.cpu().data[0].numpy()
#            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
#            gt = np.asarray(label[0].numpy(), dtype=np.int)
            output = output.transpose(1,2,0)
            # print output
            
            output = np.asarray(output[:,:,1], dtype=np.float) # output[:,:,1] is foreground ,output[:,:,0] is background
            output *= 255
            
#            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
#            output[output >0]=255
#            
            cv2.imwrite(os.path.join(args.result_dir, name[0] +'.png'),output)
#            show_all(gt, output)
#            data_list.append([gt.flatten(), output.flatten()])

    # get_iou(data_list, args.num_classes)


if __name__ == '__main__': 
    args = get_arguments()
    pprint.pprint(args)
    print('---------------------------------------------------------------' + '\n')
    RESTORE_FROM = os.path.join(args.snapshot_dir,snapPrefix+str(ITERATION_NUM)+'.pth')
    print('Saving the result of %d-th iteration model weight'%ITERATION_NUM) 
    print('The results of %s dataset is predicting '%TEST_DATASET) 
    start = timeit.default_timer()               
    main()
    end = timeit.default_timer()
    print (end-start,'seconds')
