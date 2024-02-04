from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred': # 무조건 MS 방식
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        elif flag=='pred_features':
            shuffle_flag = False; drop_last = True; batch_size = 1; freq=args.freq ###### batch를 넣어서 볼까?
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, loss_name):
        if loss_name.lower() == 'mape':
            criterion = MeanAbsolutePercentageError().to(self.device)
        else:
            criterion =  nn.MSELoss() # MAPE 사용하고 싶으면 criterion도 변경해야 함.
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        folder_path = os.path.join(self.args.checkpoints + '/figure/')
        num_epoch = self.args.train_epochs

        if self.args.inverse:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
                inverse_pred, pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark) # batch y는 true값을 얻기 위한 장치임.
                loss = criterion(inverse_pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)

                true = true.detach().cpu().numpy()
                inverse_pred = inverse_pred.detach().cpu().numpy()

                num_features = true.shape[2]

                plt.figure(figsize=(30, 20))

                for j in range(num_features):
                    file_name = f'/comparison{num_epoch}.png'
                    plt.subplot(5, 3, j+1)  # 5행 3열의 서브플롯 구성
                    plt.plot(true[:, 0, j], label='True')
                    plt.plot(inverse_pred[:, 0, j], label='Pred')
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.title(f'Feature {j+1}')
                    plt.legend()
                    plt.tight_layout()

                plt.savefig(folder_path + file_name)
                plt.close()
        else:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark) # batch y는 true값을 얻기 위한 장치임.
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)

                true = true.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()

                num_features = true.shape[2]

                plt.figure(figsize=(30, 20))

                for j in range(num_features):
                    file_name = f'/comparison{num_epoch}.png'
                    plt.subplot(5, 3, j+1)  # 5행 3열의 서브플롯 구성
                    plt.plot(true[:, 0, j], label='True')
                    plt.plot(pred[:, 0, j], label='Pred')
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.title(f'Feature {j+1}')
                    plt.legend()
                    plt.tight_layout()

                plt.savefig(folder_path + file_name)
                plt.close()

        total_loss = np.average(total_loss.cpu())
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
##########################################training##########################################################

            if self.args.inverse:
                for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    
                    model_optim.zero_grad()
                    inverse_pred, pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(inverse_pred, true)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 100==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
            else:
                for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    
                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 100==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()               

#######################################################################################################
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []

        if self.args.inverse:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader): # batch y는 true값을 얻기 위한 장치임.        
                inverse_pred, pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(inverse_pred.detach().cpu().numpy())
        else:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader): 
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark) # batch_y가 batch_x보다 1개 instance를 더 받을 수 있음.
                preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        file_name = 'comparison.png'
        plt.figure(figsize=(10, 5))
        plt.plot(trues[0, :, 0], label='True')
        plt.plot(preds[0, :, 0], label='Prediction')
        plt.title('True vs Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # 그래프 저장
        plt.savefig(folder_path + file_name)
        plt.close()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='test')
        
        if load:
            path = self.args.checkpoints
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        if self.args.inverse:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader): # batch y는 true값을 얻기 위한 장치임.        
                inverse_pred, pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred_loader.dataset.data_x[np.where(np.isnan(pred_loader.dataset.data_x).any(axis=1))[0][0]] = pred.detach().cpu().numpy()
                preds.append(inverse_pred.detach().cpu().numpy())
        else:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader): 
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark) # batch_y가 batch_x보다 1개 instance를 더 받을 수 있음.
                pred_loader.dataset.data_x[np.where(np.isnan(pred_loader.dataset.data_x).any(axis=1))[0][0]] = pred.detach().cpu().numpy() # 값 업데이트
                preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def predict_features(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred_features')
        
        if load:
            path = self.args.checkpoints
            best_model_path = path+'/'+'checkpoint.pth'
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        print(load)
        print(self.model)

        self.model.eval()
        
        preds = []
        trues = []

        if self.args.inverse:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader): # batch y는 true값을 얻기 위한 장치임.        
                inverse_pred, pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred_loader.dataset.data_x[np.where(np.isnan(pred_loader.dataset.data_x).any(axis=1))[0][0]] = pred.detach().cpu().numpy()
                preds.append(inverse_pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        else:
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader): 
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark) # batch_y가 batch_x보다 1개 instance를 더 받을 수 있음.
                pred_loader.dataset.data_x[np.where(np.isnan(pred_loader.dataset.data_x).any(axis=1))[0][0]] = pred.detach().cpu().numpy() # 값 업데이트
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # np.save(folder_path+'real_prediction.npy', preds)
        
        return preds, true

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark): # batch_x_mark / y_mark는 해당 batch_x,y의 time-stamp
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # 
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if True : #self.args.inverse:
            output = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device) # 평가를 위해서 prediction한 값과 동일한 time-stamp의 값으로 이동
        
        if output is None or 'output' not in locals(): # 만약 inverse하지 않는다면
            return outputs, batch_y
        else:
            return output, outputs, batch_y # non-scale, scale, true