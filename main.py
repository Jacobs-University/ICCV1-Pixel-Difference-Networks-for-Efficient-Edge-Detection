"""
(Training, Generating edge maps)
Pixel Difference Networks for Efficient Edge Detection

"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import models
from models.convert_pidinet import convert_pidinet
from utils import *
from edge_dataloader import BSDS_VOCLoader, Dataloader_BSDS500, Custom_Loader
from torch.utils.data import DataLoader

import torch
import torchvision


parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset')
parser.add_argument('--only-bsds', action='store_true', 
        help='only use bsds for training')
parser.add_argument('--ablation', action='store_true', 
        help='not use bsds val set for training')
parser.add_argument('--dataset', type=str, default='BSDS',
        help='data settings for BSDS dataset')

parser.add_argument('--model', type=str, default='baseline', 
        help='model to train the dataset')
parser.add_argument('--sa', action='store_true', 
        help='use CSAM in pidinet')
parser.add_argument('--dil', action='store_true', 
        help='use CDCM in pidinet')
parser.add_argument('--config', type=str, default='carv4', 
        help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None, 
        help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')
parser.add_argument('--checkinfo', action='store_true', 
        help='only check the informations about the model: model size, flops')

parser.add_argument('--epochs', type=int, default=20, 
        help='number of total epochs to run')
parser.add_argument('--iter-size', type=int, default=24, 
        help='number of samples in each iteration')
parser.add_argument('--lr', type=float, default=0.005, 
        help='initial learning rate for all weights')
parser.add_argument('--lr-type', type=str, default='multistep', 
        help='learning rate strategy [cosine, multistep]')
parser.add_argument('--lr-steps', type=str, default=None, 
        help='steps for multistep learning rate')
parser.add_argument('--opt', type=str, default='adam', 
        help='optimizer')
parser.add_argument('--wd', type=float, default=1e-4, 
        help='weight decay for all weights')
parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3, 
        help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--lmbda', type=float, default=1.1, 
        help='weight on negative pixels (the beta parameter in the paper)')

parser.add_argument('--resume', action='store_true', 
        help='use latest checkpoint if have any')
parser.add_argument('--print-freq', type=int, default=10, 
        help='print frequency')
parser.add_argument('--save-freq', type=int, default=1, 
        help='save frequency')
parser.add_argument('--evaluate', type=str, default=None, 
        help='full path to checkpoint to be evaluated')
parser.add_argument('--evaluate-converted', action='store_true', 
        help='convert the checkpoint to vanilla cnn, then evaluate')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# Training Function
def train(train_loader, model, optimizer, epoch, running_file, args, running_lr):
    batch_processing_time = AverageMeter()
    data_processing_time = AverageMeter()
    losses_temp = AverageMeter()

    ## Switch to train mode
    model.train()

    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    wD = len(str(len(train_loader)//args.iter_size))
    wE = len(str(args.epochs))

    end = time.time()
    iter_step = 0
    counter = 0
    loss_value = 0
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_loader):

        ## Measure data loading time
        data_processing_time.update(time.time() - end)

        if args.use_cuda:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        ## Compute output
        outputs = model(image)
        if not isinstance(outputs, list):
            loss = cross_entropy_loss_RCF(outputs, label, args.lmbda)
        else:
            loss = 0
            for o in outputs:
                loss += cross_entropy_loss_RCF(o, label, args.lmbda)

        counter += 1
        loss_value += loss.item()
        loss = loss / args.iter_size
        loss.backward()
        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
            iter_step += 1

            # record loss
            losses_temp.update(loss_value, args.iter_size)
            batch_processing_time.update(time.time() - end)
            end = time.time()
            loss_value = 0

            # display and logging
            if iter_step % args.print_freq == 1:
                runinfo = str(('Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}]\t' \
                          % (wE, wE, wD, wD) + \
                          'Time {batch_processing_time.val:.3f}\t' + \
                          'Data {data_processing_time.val:.3f}\t' + \
                          'Loss {loss.val:.4f} (avg:{loss.avg:.4f})\t' + \
                          'lr {lr}\t').format(
                              epoch, args.epochs, iter_step, len(train_loader)//args.iter_size,
                              batch_processing_time=batch_processing_time, data_processing_time=data_processing_time,
                              loss=losses_temp, lr=running_lr))
                print(runinfo)
                running_file.write('%s\n' % runinfo)
                running_file.flush()

    str_loss = '%.4f' % (losses_temp.avg)
    return str_loss

# Test function
def test(test_loader, model, epoch, running_file, args):

    from PIL import Image
    import scipy.io as sio
    model.eval()

    if args.ablation:
        img_dir = os.path.join(args.savedir, 'eval_results_val', 'imgs_epoch_%03d' % (epoch - 1))
        mat_dir = os.path.join(args.savedir, 'eval_results_val', 'mats_epoch_%03d' % (epoch - 1))
    else:
        img_dir = os.path.join(args.savedir, 'eval_results', 'imgs_epoch_%03d' % (epoch - 1))
        mat_dir = os.path.join(args.savedir, 'eval_results', 'mats_epoch_%03d' % (epoch - 1))
    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        print('%s already exits' % img_dir)
        #return
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    for idx, (image, img_name) in enumerate(test_loader):

        img_name = img_name[0]
        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            results = model(image)
            result = torch.squeeze(results[-1]).cpu().numpy()

        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]

        torchvision.utils.save_image(1-results_all,
                os.path.join(img_dir, "%s.jpg" % img_name))
        sio.savemat(os.path.join(mat_dir, '%s.mat' % img_name), {'img': result})
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(os.path.join(img_dir, "%s.png" % img_name))
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        print(runinfo)
        running_file.write('%s\n' % runinfo)
    running_file.write('\nDone\n')



# Main function needed to define model,use both train and test.
def main(running_file):

    global args

    ### Refine args
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()

    if args.lr_steps is not None and not isinstance(args.lr_steps, list): 
        args.lr_steps = list(map(int, args.lr_steps.split('-'))) 

    dataset_setting_choices = ['BSDS',  'Custom']
    if not isinstance(args.dataset, list): 
        assert args.dataset in dataset_setting_choices, 'unrecognized data setting %s, please choose from %s' % (str(args.dataset), str(dataset_setting_choices))
        args.dataset = list(args.dataset.strip().split('-')) 


    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    ### Output its model size, flops and bops
    if args.checkinfo:
        count_paramsM = get_model_parm_nums(model)
        print('Model size: %f MB' % count_paramsM)
        print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
        return

    ### Define optimizer
    conv_weights, bn_weights, relu_weights = model.get_weights()
    param_groups = [{
            'params': conv_weights,
            'weight_decay': args.wd,
            'lr': args.lr}, {
            'params': bn_weights,
            'weight_decay': 0.1 * args.wd,
            'lr': args.lr}, {
            'params': relu_weights, 
            'weight_decay': 0.0,
            'lr': args.lr
    }]
    info = ('conv weights: lr %.6f, wd %.6f' + \
            '\tbn weights: lr %.6f, wd %.6f' + \
            '\trelu weights: lr %.6f, wd %.6f') % \
            (args.lr, args.wd, args.lr, args.wd * 0.1, args.lr, 0.0)

    print(info)
    running_file.write('\n%s\n' % info)
    running_file.flush()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise TypeError("Please use a correct optimizer in [adam, sgd]")

    ### Transfer to cuda devices
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')


    ### Load Data
    if 'BSDS' == args.dataset[0]:
        if args.only_bsds:
            train_dataset = Dataloader_BSDS500(root=args.datadir, split="train", threshold=args.eta, ablation=args.ablation)
            test_dataset = Dataloader_BSDS500(root=args.datadir, split="test", threshold=args.eta)
        else:
            train_dataset = BSDS_VOCLoader(root=args.datadir, split="train", threshold=args.eta, ablation=args.ablation)
            test_dataset = BSDS_VOCLoader(root=args.datadir, split="test", threshold=args.eta)

    elif 'Custom' == args.dataset[0]:
        train_dataset = Custom_Loader(root=args.datadir)
        test_dataset = Custom_Loader(root=args.datadir)
    else:
        raise ValueError("unrecognized dataset setting")

    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=args.workers, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False)

    ### Create log file
    log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)

    args.start_epoch = 0
    ### Evaluate directly if required
    if args.evaluate is not None:
        checkpoint = load_checkpoint(args, running_file)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            if args.evaluate_converted:
                model.load_state_dict(convert_pidinet(checkpoint['state_dict'], args.config))
            else:
                model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError('no checkpoint loaded')
        test(test_loader, model, args.start_epoch, running_file, args)
        print('##########Time########## %s' % (time.strftime('%Y-%m-%d %H:%M:%S')))
        return

    ### Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args, running_file)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])


    ### Train
    saveID = None

    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate
        lr_str = adjust_learning_rate(optimizer, epoch, args)

        # train
        tr_avg_loss = train(
            train_loader, model, optimizer, epoch, running_file, args, lr_str)

        log = "Epoch %03d/%03d: train-loss %s | lr %s | Time %s\n" % \
              (epoch, args.epochs, tr_avg_loss, lr_str, time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(log_file, 'a') as f:
            f.write(log)

        saveID = save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, epoch, args.savedir, saveID, keep_freq=args.save_freq)

    return





if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' \
            % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))
    with open(running_file, 'w') as f:
        main(f)
    print('done')
