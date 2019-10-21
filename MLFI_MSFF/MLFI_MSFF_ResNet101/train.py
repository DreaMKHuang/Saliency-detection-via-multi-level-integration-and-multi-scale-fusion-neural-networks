from __future__ import print_function

# the libraries of System
import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
#print(gc.get_threshold())

# the libraries for Scientific Computing
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
# from torch.autograd import Variable
import torch.autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import cv2
from math import ceil

# the libraries for Data Processing
import pandas as pd

# the libraries for Auxiliary
import argparse
import pickle
import timeit
import pprint

# my libraries
from networks.MLFI_MSFF_ResNet101 import MLFI_MSFF_ResNet
from utils.datasets import ImageSaliencyDataSet
###############################################################################
# the next part is the configuration of experiments

"""
# the path of needful files
"""
TRAIN_IMAGE_DATA_DIRECTORY = '/home/hmk/Datasets/DUTS-TR/'
TRAIN_DATA_LIST = '../dataset/image/DUTS-TR.txt'
SNAPSHOT_DIR = './snapshots_MLFI_MSFF_ResNet101/'
LOG_DIR = './log_MLFI_MSFF_ResNet101/'
TRAIN_LOSS_LOG = LOG_DIR + 'train_loss_log.csv'

"""
# some Hyperparameters
"""
INPUT_SIZE = '321,321'
IMG_MEAN = np.array((101.349879883,118.173160985,125.531285164), dtype=np.float32)
BATCH_SIZE = 4
LEARNING_RATE = 1.0e-4 # 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 81000
POWER = 0.9
WEIGHT_DECAY = 0.0005
SAVE_PRED_EVERY = 2700
RESTORE_FROM = '../model/pre_trained_model/MS_DeepLab_resnet_pretrained_COCO_init.pth'
RANDOM_ROTATE = True
RANDOM_CROP = True
snapPrefix = 'image_saliency_'
#device_ids = [0,1]
NUM_WORKERS = 4


###############################################################################
# let's prepare for training the model!
        
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MLFI_MSFF_Network")
    """
    # Hyperparameters
    """
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    """
    # path
    """
    parser.add_argument("--train-image-data-dir", type=str, default=TRAIN_IMAGE_DATA_DIRECTORY,
                        help="Path to the directory containing the DAVIS dataset.")
    parser.add_argument("--train-data-list", type=str, default=TRAIN_DATA_LIST,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--train-loss-log", type=str, default=TRAIN_LOSS_LOG,
                        help="Path to the directory containing the train loss log file.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Where to save log of training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")

    """
    # other parameters
    """
    parser.add_argument("--random-rotate", default=RANDOM_ROTATE,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-crop", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS, 
                        help='number of data loading workers')
    return parser.parse_args()

###############################################################################
# OK! Let's train the model.
def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False, therefore this function does not return 
    any batchnorm parameter
    """
    b = []
    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.smooth_block1.parameters())
    b.append(model.smooth_block2.parameters())
    b.append(model.smooth_block3.parameters())
    b.append(model.latlayer1.parameters())
    b.append(model.latlayer2.parameters())
    b.append(model.latlayer3.parameters())
    b.append(model.latlayer4.parameters())
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
                       
def adjust_learning_rate(args, optimizer, i_iter):
    # Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs
#    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    lr = args.learning_rate*((1-float(i_iter)/args.num_steps)**(args.power))
#    print('Iteration [{}] Learning rate: {}'.format(i_iter, lr))
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def main():
    """Create the model and start the training."""
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    cudnn.benchmark = True
    # Create network.
    model = MLFI_MSFF_ResNet(num_classes=args.num_classes)
#    print model.state_dict().keys()    
    new_params = model.state_dict().copy()
    saved_state_dict = torch.load(args.restore_from)
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not i_parts[1] =='layer5':
            # print saved_state_dict[i].size()
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
        
    model.train()
    model.to(torch.device('cuda', args.gpu))
    trainloader = data.DataLoader(ImageSaliencyDataSet(
            args.train_image_data_dir,
            args.train_data_list, 
            max_iters=args.num_steps*args.batch_size, 
            input_size=input_size, 
            crop=args.random_crop, 
            rotate=args.random_rotate, 
            mean=IMG_MEAN,
            ), 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True)        
    optimizer = optim.SGD([
            {
                    'params': get_1x_lr_params(model), 
                    'lr': args.learning_rate 
                    }, 
            {
                    'params': get_10x_lr_params(model), 
                    'lr': 10*args.learning_rate
                    }], 
                lr=args.learning_rate, 
                momentum=args.momentum,
                weight_decay=args.weight_decay)
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    for i_iter, batch in enumerate(trainloader):
        image, label, _, name = batch

        image = image.to(torch.device('cuda', args.gpu))
        label = label.type(torch.LongTensor).to(torch.device('cuda', args.gpu))

        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)
        pred = nn.functional.interpolate(model(image), size=input_size, mode='bilinear', align_corners=True)
        # print(pred)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        print ('iter = ', i_iter, 'of', args.num_steps,'completed, loss = ', loss.item())
        
        trainlosslog = pd.DataFrame({
            'train_loss':[loss.item()]})
        if i_iter == 0:
            trainlosslog.to_csv(args.train_loss_log, mode='a', index=False, sep=',')
        else:
            trainlosslog.to_csv(args.train_loss_log, mode='a', header=False, index=False, sep=',')

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),os.path.join(args.snapshot_dir, snapPrefix+str(int(i_iter/args.save_pred_every))+'.pth'))   
        elif i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),os.path.join(args.snapshot_dir, snapPrefix+str(int(args.num_steps/args.save_pred_every))+'.pth'))
            break
        gc.collect()

if __name__ == '__main__':
    args = get_arguments()
    pprint.pprint(args)
    print('---------------------------------------------------------------' + '\n')
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.train_loss_log):
        os.mknod(args.train_loss_log)
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print (end-start,'seconds')


