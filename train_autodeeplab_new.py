import argparse
import os
import sys
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from mypath import Path
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from model_search import AutoDeeplab
from architect import Architect

class Trainer(object):
    def __init__(self, args):
        def _build_opt():
            if args.mode == 0:
                self.opt = torch.optim.SGD(
                    self.model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd
                )
                self.opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt,
                    float(args.epochs),
                    eta_min=args.m_lr
                )
            elif args.mode in [1, 2, 3]:
                alpha_sign = 'w_alpha'
                beta_sign = 'w_beta'
                self.w_opt = torch.optim.SGD(
                    [p[1] for p in self.model.named_parameters() if (p[0].find(alpha_sign) < 0 and p[0].find(beta_sign) < 0)],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd
                )
                self.alpha_opt = torch.optim.Adam(
                    [p[1] for p in self.model.named_parameters() if p[0].find(alpha_sign) >= 0],
                    lr=args.alpha_lr,
                    betas=(0.5, 0.999),
                    weight_decay=args.alpha_wd
                )
                self.beta_opt = torch.optim.Adam(
                    [p[1] for p in self.model.named_parameters() if p[0].find(beta_sign) >= 0],
                    lr=args.beta_lr,
                    betas=(0.5, 0.999),
                    weight_decay=args.beta_wd
                )

                self.opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.w_opt,
                    float(args.epochs),
                    eta_min=args.m_lr
                )
            else:
                print('invalid search mode.')
                sys.exit(0)

        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader1, self.train_loader2, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                #if so, which trainloader to use?
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define network and lr scheduler
        self.model = AutoDeeplab(num_classes=self.nclass, num_layers=12, criterion=self.criterion)
        _build_opt()

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        ## Define lr scheduler
        #self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                                    args.epochs, len(self.train_loader1))
        #
        #self.architect = Architect(self.model, args)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model.cuda())
            self.model = self.model.cuda()
            print('cuda finished')

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def train_0(self, epoch):
        total_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader1)
        num_img_tr = len(self.train_loader1)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            output = self.model(image)
            loss = self.criterion(output, target)
            self.opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.opt.step()
            total_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (total_loss / (i + 1)))
            #self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

            #torch.cuda.empty_cache()
        self.writer.add_scalar('train/total_loss_epoch', total_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % total_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def train_1(self, epoch):
        total_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader1)
        num_img_tr = len(self.train_loader1)
        for i, sample in enumerate(tbar):
            image_1, target_1 = sample['image'], sample['label']
            if self.args.cuda:
                image_1, target_1 = image_1.cuda(), target_1.cuda()
            output_1 = self.model(image_1)
            loss_1 = self.criterion(output_1, target_1)
            self.w_opt.zero_grad()
            loss_1.backward()
            total_loss += loss_1.item()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)  # clip gradient
            self.w_opt.step()

            if epoch >= self.args.ab_epoch:
                try:
                    (image_2, target_2) = self.train_loader2.next()
                except:
                    train_loader2 = iter(self.train_loader2)
                    (image_2, target_2) = train_loader2.next()
                if self.args.cuda:
                    image_2, target_2 = image_2.cuda(), target_2.cuda()
                output_2 = self.model(image_2)
                loss_2 = self.criterion(output_2, target_2)
                self.a_opt.zero_grad()
                self.b_opt.zero_grad()
                loss_2.backward()
                self.a_opt.step()
                self.b_opt.step()

            tbar.set_description('Train loss: %.3f' % (total_loss / (i + 1)))
            #self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image_1, target_1, output_1, global_step)

            #torch.cuda.empty_cache()
        self.writer.add_scalar('train/total_loss_epoch', total_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image_1.data.shape[0]))
        print('Loss: %.3f' % total_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def train_2(self, epoch):
        p_alpha = [p[1] for p in self.model.named_parameters() if p[0].find('w_alpha') >= 0]
        p_beta  = [p[1] for p in self.model.named_parameters() if p[0].find('w_beta')  >= 0]
        total_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader1)
        num_img_tr = len(self.train_loader1)
        for i, sample in enumerate(tbar):
            image_1, target_1 = sample['image'], sample['label']
            try:
                (image_2, target_2) = self.train_loader2.next()
            except:
                train_loader2 = iter(self.train_loader2)
                (image_2, target_2) = train_loader2.next()
            if self.args.cuda:
                image_1, target_1 = image_1.cuda(), target_1.cuda()
                image_2, target_2 = image_2.cuda(), target_2.cuda()

            output_2 = self.model(image_2)
            loss_2 = self.criterion(output_2, target_2)
            grad_alpha = [grad.detach().clone() for grad in torch.autograd.grad(loss_2, p_alpha)]
            grad_beta  = [grad.detach().clone() for grad in torch.autograd.grad(loss_2, p_beta )]

            output_1 = self.model(image_1)
            loss_1 = self.criterion(output_1, target_1)
            self.w_opt.zero_grad()
            loss_1.backward()
            total_loss += loss_1.item()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)  # clip gradient
            self.w_opt.step()
            tbar.set_description('Total loss: %.3f' % (total_loss / (i + 1)))
            # self.writer.add_scalar('train/total_loss_iter', loss_1.item(), i + num_img_tr * epoch)

            # update alpha & beta
            if epoch >= self.args.ab_epoch:
                for (p, v) in zip(p_alpha, grad_alpha):
                    p.grad.copy_(v)
                self.a_opt.step()
                for (p, v) in zip(p_beta, grad_beta):
                    p.grad.copy_(v)
                self.b_opt.step()

            total_loss += loss_1.item()
            tbar.set_description('Train loss: %.3f' % (total_loss / (i + 1)))
            # self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image_1, target_1, output_1, global_step)

            # torch.cuda.empty_cache()
        self.writer.add_scalar('train/total_loss_epoch', total_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image_1.data.shape[0]))
        print('Loss: %.3f' % total_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def train_3(self, epoch):
        p_alpha = [p[1] for p in self.model.named_parameters() if p[0].find('w_alpha') >= 0]
        p_beta = [p[1] for p in self.model.named_parameters() if p[0].find('w_beta') >= 0]
        p_convs = [p[1] for p in self.model.named_parameters() if (p[0].find('w_alpha') < 0 and p[0].find('w_beta') < 0)]
        def _get_weight_():
            return [p.data.clone() for p in p_convs]

        def _get_alpha_():
            return [p.data.clone() for p in p_alpha]

        def _get_beta_():
            return [p.data.clone() for p in p_beta]

        def _set_weight_(value):
            for (v, p) in zip(value, p_convs):
                p.data.copy_(v)

        def _set_alpha_(value):
            for (v, p) in zip(value, p_alpha):
                p.data.copy_(v)

        def _set_beta_(value):
            for (v, p) in zip(value, p_beta):
                p.data.copy_(v)

        def _set_weight_grad_(grads):
            for (p, g) in zip(p_convs, grads):
                p.grad.copy_(g)

        def _set_alpha_grad_(grads):
            for (p, g) in zip(p_alpha, grads):
                try:
                    p.grad.copy_(g)
                except:
                    p.grad = g.clone()

        def _set_beta_grad_(grads):
            for (p, g) in zip(p_beta, grads):
                try:
                    p.grad.copy_(g)
                except:
                    p.grad = g.clone()

        self.model.train()
        total_loss = 0.0
        self.sche.step()
        tbar = tqdm(self.train_loader1)
        num_img_tr = len(self.train_loader1)
        for i, sample in enumerate(tbar):
            image_1, target_1 = sample['image'], sample['label']
            try:
                (image_2, target_2) = self.train_loader2.next()
            except:
                train_loader2 = iter(self.train_loader2)
                (image_2, target_2) = train_loader2.next()
            if self.args.cuda:
                image_1, target_1 = image_1.cuda(), target_1.cuda()
                image_2, target_2 = image_2.cuda(), target_2.cuda()

            if epoch >= self.args.ab_epoch:
                w = _get_weight_()
                output_1 = self.model(image_1)
                loss_1 = self.criterion(output_1, target_1)
                g_ = torch.autograd.grad(loss_1, p_convs)
                try:
                    g_w = [(g + self.args.momentum * self.w_opt.state[p]).data.clone() for (g, p) in zip(g_, p_convs)]
                except:
                    g_w = [g.data.clone() for g in g_]

                w_t = [v - g * self.args.lr for (v, g) in zip(w, g_w)]

                _set_weight_(w_t)
                output_2 = self.model(image_2)
                loss_2 = self.criterion(output_2, target_2)
                g_a_l = [g_.data.clone() for g_ in torch.autograd.grad(loss_2, p_alpha)]
                g_b_l = [g_.data.clone() for g_ in torch.autograd.grad(loss_2, p_beta)]
                g_w_t = [g_.data.clone() for g_ in torch.autograd.grad(loss_2, p_convs)]

                R = 0.01 / math.sqrt(sum((w_ * w_).sum() for w_ in w_t))

                w_n = [w_ - R * g_w for (w_, g_w_t_) in zip(w, g_w_t)]
                w_p = [w_ + R * g_w for (w_, g_w_t_) in zip(w, g_w_t)]

                _set_weight_(w_n)
                output_1 = self.model(image_1)
                loss_1 = self.criterion(output_1, target_1)
                g_a_n = [g_a.data.clone() for g_a in torch.autograd.grad(loss_1, p_alpha)]
                g_b_n = [g_b.data.clone() for g_b in torch.autograd.grad(loss_1, p_beta)]

                _set_weight_(w_p)
                output_1 = self.model(image_1)
                loss_1 = self.criterion(output_1, target_1)
                g_a_p = [g_a.data.clone() for g_a in torch.autograd.grad(loss_1, p_alpha)]
                g_b_p = [g_b.data.clone() for g_b in torch.autograd.grad(loss_1, p_beta)]

                g_a_r = [(gr - gl) / (2 * R) for (gr, gl) in zip(g_a_p, g_a_n)]
                g_b_r = [(gr - gl) / (2 * R) for (gr, gl) in zip(g_b_p, g_b_n)]

                g_a = [gl - self.args.lr * gr for (gl, gr) in zip(g_a_l, g_a_r)]
                g_b = [gl - self.args.lr * gr for (gl, gr) in zip(g_b_l, g_b_r)]

                _set_alpha_grad_(g_a)
                self.a_opt.step()
                _set_beta_grad_(g_b)
                self.b_opt.step()
                _set_weight_(w)

            output_1 = self.model(image_1)
            loss_1 = self.criterion(output_1, target_1)
            self.w_opt.zero_grad()
            loss_1.backward()
            total_loss += loss_1.item()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)  # clip gradient
            self.w_opt.step()
            tbar.set_description('Train loss: %.3f' % (total_loss / (i + 1)))
            # self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image_1, target_1, output_1, global_step)

            # torch.cuda.empty_cache()
        self.writer.add_scalar('train/total_loss_epoch', total_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image_1.data.shape[0]))
        print('Loss: %.3f' % total_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def train(self, epoch):
        if self.args.mode == 0:
            self.train_0(epoch)
        elif self.args.mode == 1:
            self.train_1(epoch)
        elif self.args.mode == 2:
            self.train_2(epoch)
        elif self.args.mode == 3:
            self.train_3(epoch)
        else:
            print('invalid mode.')
            sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="AutoDeeplab")
    parser.add_argument('--out_stride', type=int, default=16, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'cityscapes', 'kd'], help='dataset name (default: pascal)')
    parser.add_argument('--use_sbd', action='store_true', default=False, help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=224, help='base image size')
    parser.add_argument('--crop_size', type=int, default=224, help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--mode', type=int, default=1, help='the training mode')
    parser.add_argument('--epochs', type=int, default=None, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--ab_epoch', type=int, default=5, metavar='N', help='epoch to start training alphas')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=None, metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False, help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='learning rate (default: auto)')
    #parser.add_argument('--arch_lr', type=float, default=1e-3, metavar='LR', help='architect learning rate (default: auto)')
    parser.add_argument('--alpha_lr', type=float, default=0.003, help='learning rate for alpha')
    parser.add_argument('--beta_lr', type=float, default=0.003, help='learning rate for beta')
    parser.add_argument('--m_lr', type=float, default=0.001, help='min learning rate')
    #parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--alpha_wd', type=float, default=1e-3, help='alpha weight decay')
    parser.add_argument('--beta_wd', type=float, default=1e-3, help='beta weight decay')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    #parser.add_argument('--arch_weight_decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'kd': 10
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 2 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'AutoDeeplab'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.train(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
   main()
