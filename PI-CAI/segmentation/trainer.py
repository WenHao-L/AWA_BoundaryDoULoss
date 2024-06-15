import math
import os
import shutil
import warnings

import numpy as np
import torch
from picai_eval import evaluate
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from segmentation.data_loader import (DataGenerator, Normalize, RandomFlip2D,
                                      RandomRotate2D, To_Tensor)
from segmentation.data_loader_new import (DataGenerator_new, Normalize_new, RandomFlip2D_new,
                                          RandomRotate2D_new, To_Tensor_new)
from segmentation.loss import Deep_Supervised_Loss, BDoU_Deep_Supervised_Loss, Tversky_Deep_Supervised_Loss, Boundary_Deep_Supervised_Loss
from segmentation.model import itunet_2d, aw_itunet_2d, wso_itunet_2d, swinunetr_2d, aw_swinunetr_2d, unetr_2d, aw_unetr_2d, unet_2d, aw_unet_2d
from segmentation.utils import dfs_remove_weight, poly_lr

warnings.filterwarnings('ignore')


class SemanticSeg(object):
    def __init__(self,lr=1e-3,n_epoch=1,channels=3,num_classes=2, input_shape=(384,384),batch_size=6,num_workers=0,
                  device=None,pre_trained=False,ckpt_point=True,weight_path=None,weight_decay=0.0001,
                  use_fp16=False,transformer_depth = 18, use_aw=False, use_checkpoint=False,
                  use_wso=False, act_window='relu', upbound_window=1., window_init=None, model="itunet_d24"
                ):
        super(SemanticSeg,self).__init__()
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.use_fp16 = use_fp16

        self.pre_trained = pre_trained
        self.ckpt_point = ckpt_point
        self.weight_path = weight_path
        self.weight_decay = weight_decay

        self.start_epoch = 0
        self.global_step = 0
        self.metrics_threshold = 0.

        self.transformer_depth = transformer_depth

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        if use_aw:
            if model == "itunet_d24":
                self.net = aw_itunet_2d(n_channels=self.channels,n_classes=self.num_classes, image_size= tuple(self.input_shape), transformer_depth = self.transformer_depth, use_checkpoint=use_checkpoint)
            elif model == "swinunetr_d24":
                self.net = aw_swinunetr_2d()
            elif model == "unetr_d24":
                self.net = aw_unetr_2d()
            elif model == "unet_d24":
                self.net = aw_unet_2d(n_channels=self.channels,n_classes=self.num_classes,use_checkpoint=use_checkpoint)
            else:
                raise NotImplementedError
        elif use_wso:
            if model == "itunet_d24":
                self.net = wso_itunet_2d(n_channels=self.channels,n_classes=self.num_classes, image_size= tuple(self.input_shape), transformer_depth = self.transformer_depth, act_window=act_window, upbound_window=upbound_window, window_init=window_init)
            else:
                raise NotImplementedError
        else:
            if model == "itunet_d24":
                self.net = itunet_2d(n_channels=self.channels,n_classes=self.num_classes, image_size= tuple(self.input_shape), transformer_depth = self.transformer_depth)
            elif model == "swinunetr_d24":
                self.net = swinunetr_2d()
            elif model == "unetr_d24":
                self.net = unetr_2d()
            elif model == "unet_d24":
                self.net = unet_2d(n_channels=self.channels,n_classes=self.num_classes)
            else:
                raise NotImplementedError
            
        if self.pre_trained:
            self._get_pre_trained(self.weight_path,ckpt_point)

        if self.channels == 5:
            self.train_transform = [
                Normalize_new(),   #1
                RandomRotate2D_new(),  #6
                RandomFlip2D_new(mode='hv'),  #7
                To_Tensor_new(num_class=self.num_classes,input_channel = 3)   # 10
            ]
        else:
            self.train_transform = [
                Normalize(),   #1
                RandomRotate2D(),  #6
                RandomFlip2D(mode='hv'),  #7
                To_Tensor(num_class=self.num_classes,input_channel = self.channels)   # 10
            ]

    def trainer(self,train_path,val_path,val_ap, cur_fold, tz_2d_dir, tz_3d_dir, pz_2d_dir, pz_3d_dir, output_dir=None,log_dir=None,phase = 'seg',loss=None):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold"+str(cur_fold))
        log_dir = os.path.join(log_dir, "fold"+str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)
            
        self.step_pre_epoch = len(train_path) // self.batch_size
        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(len(train_path)/self.batch_size)

        net = self.net
        lr = self.lr

        if loss == "BoundaryDoULoss":
            loss = BDoU_Deep_Supervised_Loss()
        elif loss == "TverskyLoss":
            loss = Tversky_Deep_Supervised_Loss()
        elif loss == "BoundaryLoss":
            loss = Boundary_Deep_Supervised_Loss()
        else:
            loss = Deep_Supervised_Loss()

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        train_transformer = transforms.Compose(self.train_transform)

        if self.channels == 5:
            train_dataset = DataGenerator_new(train_path, tz_dir=tz_2d_dir, pz_dir=pz_2d_dir, num_class=self.num_classes,transform=train_transformer)
        else:
            train_dataset = DataGenerator(train_path,num_class=self.num_classes,transform=train_transformer)

        train_loader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=self.num_workers,
          pin_memory=True
        )

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=self.weight_decay)  # origin
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        # optimizer = torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=self.weight_decay)

        scaler = GradScaler()

        early_stopping = EarlyStopping(patience=160,verbose=True,monitor='val_score',op_type='max')

        epoch = self.start_epoch
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr = lr)  # origin
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        while epoch < self.n_epoch:
            train_loss,train_dice,train_run_dice = self._train_on_epoch(epoch,net,loss,optimizer,train_loader,scaler)
            # scheduler.step()  # CosineAnnealingLR

            if phase == 'seg':
                # val_loss,val_dice,val_run_dice = self._val_on_epoch(epoch,net,loss,val_path,tz_2d_dir, pz_2d_dir)
                # score = val_run_dice
                val_loss,val_dice,val_run_dice = self._val_on_epoch_new(epoch,net,loss,val_ap,tz_3d_dir, pz_3d_dir)
                score = val_dice
                # scheduler.step(val_loss)  # ReduceLROnPlateau
            else:
                ap = self.val(val_ap,net,mode = 'train')
                score = ap.score + 0.02*train_run_dice

            optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr = lr)  # origin

            torch.cuda.empty_cache()
                
            self.writer.add_scalar(
              'data/lr',optimizer.param_groups[0]['lr'],epoch
            )

            early_stopping(score)

            #save
            if score > self.metrics_threshold:
                self.metrics_threshold = score

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                  'epoch':epoch,
                  'save_dir':output_dir,
                  'state_dict':state_dict,
                }
                
                if phase == 'seg':
                    file_name = 'epoch:{}-train_loss:{:.5f}-train_dice:{:.5f}-train_run_dice:{:.5f}-val_loss:{:.5f}-val_dice:{:.5f}-val_run_dice:{:.5f}.pth'.format(
                    epoch,train_loss,train_dice,train_run_dice,val_loss,val_dice,val_run_dice) 
                else:
                    file_name = 'epoch:{}-train_loss:{:.5f}-train_dice:{:.5f}-train_run_dice:{:.5f}-val_auroc:{:.5f}-val_ap:{:.5f}-val_score:{:.5f}.pth'.format(
                    epoch,train_loss,train_dice,train_run_dice,ap.auroc,ap.AP,ap.score)
                save_path = os.path.join(output_dir,file_name)
                print("Save as: %s" % file_name)

                torch.save(saver,save_path)
            
            epoch += 1
            
            # early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(output_dir,retain=3)

    def _train_on_epoch(self,epoch,net,criterion,optimizer,train_loader,scaler):
        net.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()

        from segmentation.metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        for step,sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            with autocast(self.use_fp16):
                output = net(data)
                if isinstance(output,tuple):
                    output = output[0]
                loss = criterion(output,target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = output[0]
                       
            output = output.float()
            loss = loss.float()

            dice = compute_dice(output.detach(),target)
            train_loss.update(loss.item(),data.size(0))
            train_dice.update(dice.item(),data.size(0))
            
            output = torch.argmax(torch.softmax(output,dim=1),1).detach().cpu().numpy()  #N*H*W 
            target = torch.argmax(target,1).detach().cpu().numpy()
            run_dice.update_matrix(target,output)

            torch.cuda.empty_cache()

            if self.global_step%10==0:
                rundice, dice_list = run_dice.compute_dice() 
                print("Category Dice: ", dice_list)
                print('epoch:{}/{},step:{},train_loss:{:.5f},train_dice:{:.5f},run_dice:{:.5f},lr:{}'.format(epoch,self.n_epoch, step, loss.item(), dice.item(), rundice, optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                self.writer.add_scalars(
                  'data/train_loss_dice',{'train_loss':loss.item(),'train_dice':dice.item()},self.global_step
                )

            self.global_step += 1

        return train_loss.avg,train_dice.avg,run_dice.compute_dice()[0]


    def _val_on_epoch(self,epoch,net,criterion,val_path,tz_2d_dir=None, pz_2d_dir=None,val_transformer=None):
        net.eval()

        if self.channels == 5:
            val_transformer = transforms.Compose([
                Normalize_new(), 
                To_Tensor_new(num_class=self.num_classes,input_channel = 3)
            ])
            val_dataset = DataGenerator_new(val_path,num_class=self.num_classes,transform=val_transformer,tz_dir=tz_2d_dir,pz_dir=pz_2d_dir)

        else:
            val_transformer = transforms.Compose([
                Normalize(), 
                To_Tensor(num_class=self.num_classes,input_channel = self.channels)
            ])
            val_dataset = DataGenerator(val_path,num_class=self.num_classes,transform=val_transformer)

        val_loader = DataLoader(
          val_dataset,
          batch_size=self.batch_size,
          shuffle=False,
          num_workers=self.num_workers,
          pin_memory=True
        )

        val_loss = AverageMeter()
        val_dice = AverageMeter()

        from segmentation.metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        with torch.no_grad():
            for step,sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]
                loss = criterion(output,target)

                output = output[0]

                output = output.float()
                loss = loss.float()

                dice = compute_dice(output.detach(),target)
                val_loss.update(loss.item(),data.size(0))
                val_dice.update(dice.item(),data.size(0))

                output = torch.softmax(output,dim=1)

                output = torch.argmax(output,1).detach().cpu().numpy()  #N*H*W 
                target = torch.argmax(target,1).detach().cpu().numpy()
                run_dice.update_matrix(target,output)

                torch.cuda.empty_cache()

                if step % 10 == 0:
                    rundice, dice_list = run_dice.compute_dice() 
                    print("Category Dice: ", dice_list)
                    print('epoch:{}/{},step:{},val_loss:{:.5f},val_dice:{:.5f},run_dice:{:.5f}'.format(epoch,self.n_epoch, step, loss.item(), dice.item(), rundice))
                    # run_dice.init_op()

        return val_loss.avg,val_dice.avg,run_dice.compute_dice()[0]


    def _val_on_epoch_new(self,epoch,net,criterion,val_path,tz_3d_dir=None, pz_3d_dir=None,val_transformer=None):
        net.eval()

        class Normalize_2d(object):
            def __call__(self,sample):
                ct = sample['ct']
                seg = sample['seg']
                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        if np.max(ct[i,j])!=0:
                            ct[i,j] = ct[i,j]/np.max(ct[i,j])
                    
                new_sample = {'ct':ct, 'seg':seg}
                return new_sample
        
        class Normalize_2d_new(object):
            def __call__(self,sample):
                ct = sample['ct']
                seg = sample['seg']
                tz = sample['tz']
                pz = sample['pz']

                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        if np.max(ct[i,j])!=0:
                            ct[i,j] = ct[i,j]/np.max(ct[i,j])
                    
                new_sample = {'ct':ct, 'seg':seg, 'tz':tz, 'pz':pz}
                return new_sample

        if self.channels == 5:
            val_transformer = transforms.Compose([Normalize_2d_new(),To_Tensor_new(num_class=self.num_classes,input_channel = 3)])
            val_dataset = DataGenerator_new(val_path,num_class=self.num_classes,transform=val_transformer,tz_dir=tz_3d_dir, pz_dir=pz_3d_dir)

        else:
            val_transformer = transforms.Compose([Normalize_2d(),To_Tensor(num_class=self.num_classes,input_channel = self.channels)])
            val_dataset = DataGenerator(val_path,num_class=self.num_classes,transform=val_transformer)

        val_loader = DataLoader(
          val_dataset,
          batch_size=1,
          shuffle=False,
          num_workers=self.num_workers,
          pin_memory=True
        )

        val_loss = AverageMeter()
        val_dice = AverageMeter()

        from segmentation.metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        with torch.no_grad():
            for step,sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']
                
                data = data.squeeze().transpose(1,0) 
                data = data.cuda()
                target = target.squeeze().transpose(1,0) 
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]
                loss = criterion(output,target)

                output = output[0]

                output = output.float()
                loss = loss.float()

                dice = compute_dice_new(output.detach(),target)
                val_loss.update(loss.item(),data.size(0))
                val_dice.update(dice.item(),data.size(0))

                output = torch.softmax(output,dim=1)

                output = torch.argmax(output,1).detach().cpu().numpy()  #N*H*W 
                target = torch.argmax(target,1).detach().cpu().numpy()
                run_dice.update_matrix(target,output)

                torch.cuda.empty_cache()

                if step % 10 == 0:
                    rundice, dice_list = run_dice.compute_dice() 
                    print("Category Dice: ", dice_list)
                    print('epoch:{}/{},step:{},val_loss:{:.5f},val_dice:{:.5f},run_dice:{:.5f}'.format(epoch,self.n_epoch, step, loss.item(), dice.item(), rundice))
                    # run_dice.init_op()

        self.writer.add_scalars(
            'data/val_loss_dice', {'val_loss':val_loss.avg, 'val_dice':val_dice.avg, 'run_dice':run_dice.compute_dice()[0]},epoch
        )

        return val_loss.avg,val_dice.avg,run_dice.compute_dice()[0]


    def val(self,val_path,net = None,val_transformer=None,mode = 'val'):
        if net is None:
            net = self.net
            net = net.cuda()
        net.eval()

        class Normalize_2d(object):
            def __call__(self,sample):
                ct = sample['ct']
                seg = sample['seg']
                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        if np.max(ct[i,j])!=0:
                            ct[i,j] = ct[i,j]/np.max(ct[i,j])
                    
                new_sample = {'ct':ct, 'seg':seg}
                return new_sample

        val_transformer = transforms.Compose([Normalize_2d(),To_Tensor(num_class=self.num_classes,input_channel = self.channels)])

        if self.channels == 5:
            val_dataset = DataGenerator_new(val_path,num_class=self.num_classes,transform=val_transformer)
        else:
            val_dataset = DataGenerator(val_path,num_class=self.num_classes,transform=val_transformer)

        val_loader = DataLoader(
          val_dataset,
          batch_size=1,
          shuffle=False,
          num_workers=self.num_workers,
          pin_memory=True
        )

        y_pred = []
        y_true = []

        with torch.no_grad():
            for step,sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']
                
                data = data.squeeze().transpose(1,0) 
                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]

                output = output[0]

                output = output.float()
                output = torch.softmax(output,dim=1)  #N*H*W 
                output = output[:,0,:,:]
                output = 1-output
                output = output.detach().cpu().numpy()

                from report_guided_annotation import extract_lesion_candidates

                # process softmax prediction to detection map
                if mode == 'train':
                    cspca_det_map_npy = extract_lesion_candidates(
                        output, threshold='dynamic-fast')[0]
                else:
                    cspca_det_map_npy = extract_lesion_candidates(
                        output, threshold='dynamic',num_lesions_to_extract=5,min_voxels_detection=10,dynamic_threshold_factor = 2.5)[0]

                # remove (some) secondary concentric/ring detections
                cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/2)] = 0

                y_pred.append(cspca_det_map_npy)
                target = torch.argmax(target,1).detach().cpu().numpy().squeeze()
                target[target>0] = 1
                y_true.append(target)

                print(np.sum(target)>0,np.max(cspca_det_map_npy))

                torch.cuda.empty_cache()
                # break
        m = evaluate(y_pred,y_true)
        print(m)
        return m


    def _get_pre_trained(self,weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_loss',op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
           print(self.monitor, f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.val_score_min = val_score

class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def binary_dice(predict, target, smooth=1e-5):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1) #N，H*W
    target = target.contiguous().view(target.shape[0], -1) #N，H*W

    inter = torch.sum(torch.mul(predict, target), dim=1) #N
    union = torch.sum(predict + target, dim=1) #N

    dice = (2*inter + smooth) / (union + smooth ) #N

    return dice.mean()

def compute_dice(predict,target,ignore_index=0):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)
    
    onehot_predict = torch.argmax(predict,dim=1)#N*H*W
    onehot_target = torch.argmax(target,dim=1) #N*H*W

    dice_list = np.ones((target.shape[1]),dtype=np.float32)
    for i in range(target.shape[1]):
        if i != ignore_index:
            if i not in onehot_predict and i not in onehot_target:
                continue
            dice = binary_dice((onehot_predict==i).float(), (onehot_target==i).float())
            dice_list[i] = round(dice.item(),4)
    
    return np.nanmean(dice_list[1:])

def binary_dice_new(predict, target, smooth=1e-5):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(-1) #N*H*W
    target = target.contiguous().view(-1) #N*H*W

    inter = torch.sum(torch.mul(predict, target)) #N
    union = torch.sum(predict + target) #N

    dice = (2*inter + smooth) / (union + smooth ) #N

    return dice

def compute_dice_new(predict,target,ignore_index=0):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)
    
    onehot_predict = torch.argmax(predict,dim=1)#N*H*W
    onehot_target = torch.argmax(target,dim=1) #N*H*W

    dice_list = np.ones((target.shape[1]),dtype=np.float32)
    for i in range(target.shape[1]):
        if i != ignore_index:
            if i not in onehot_predict and i not in onehot_target:
                continue
            dice = binary_dice_new((onehot_predict==i).float(), (onehot_target==i).float())
            dice_list[i] = round(dice.item(),4)
    
    return np.nanmean(dice_list[1:])