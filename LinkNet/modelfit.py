"""
The module modelfit is used to train the model.

Author: Rayykia

Typical usage sample:

>>> from modelfit import Fit
>>> train_model = Fit(
        epoches=EPOCHES,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        model=model,
        loss_fn='CrossEntropyLoss',
        optim=optimizer,
        lr_step_size=5,
        lr_decay_gamma=0.9
    )
>>> trained_model = train_model.fit()
>>> train_model.plot()

Historical Version:

8/20/2023:  Fit.train() y = torch.tensor(y, dtype=torch.float)
8/28/2023:  Save the best parameters among all epoches to MODEL_PATH.
8/17/2023:  The learning rate decay is added to the training process.
8/28/2023:  Add MSELoss.
            Add calss LocalizationFit().
8/29/2023:  Add class SemSegFit().
8/30/2023:  Add IoU index to semantic segmentation fit model.

"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import copy
import os
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_(dataloader_: DataLoader, 
          model: nn.Module, 
          loss_fn, optimizer: torch.optim.Adam) -> [float, float]:
    '''
    Train the parameters of the model for 1 epoch.

    Parameters:
    :dataloader_    : The train dataloader.
    :model          : The model whose paranmeters are trained.
    :loss_fn        : The chosen loss function. (nn.CrossEntropyLoss, ...)
    :optimizer      : The optimizer used. (Adam, SGD, ...)

    Returns:
    :acc            : The accuracy of the model of this epoch on the train dataset.
    :avg_train_loss : The average train loss of each batch.
    '''
    size = len(dataloader_.dataset)
    num_batches = len(dataloader_)

    # train_loss: Accumulate loss from all batches.
    # correct: Accumulate the correct samples.
    train_loss, correct = 0, 0

    for x, y in dataloader_:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)  # logits
        loss = loss_fn(pred, y)

        # core code
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            train_loss += loss.item()
        
    acc = correct / size
    avg_train_loss = train_loss / num_batches
    return acc, avg_train_loss


def test_(test_dataloader: DataLoader, 
         model: nn.Module, 
         loss_fn) -> [float, float]:
    '''
    Test the model on the test dataset (packaged with dataloader).

    Parameters:
    :test_dataloader    : The test dataloader.
    :model              : The model whose paranmeters are trained.
    :loss_fn            : The chosen loss function. (nn.CrossEntropyLoss, ...)

    Returns:
    :test_acc           : The accuracy of the model of this epoch on the test dataset.
    :avg_test_loss      : The average train loss of each batch.
    '''
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    test_loss, test_correct = 0, 0 

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss += loss.item()

        test_acc = test_correct / size
        avg_test_loss = test_loss / num_batches
    return test_acc, avg_test_loss




class ClassifierFit():
    '''
    Train the given model.

    Attributes:
    :epoches            : The epoches trained.
    :train_dataloader   : The train dataloader.
    :test_dataloader    : The test dataloader
    :model              : The model whose paranmeters are trained.
    :loss_fn            : The chosen loss function. (nn.CrossEntropyLoss, ...)
    :opt                : The name of the optimizer used. (Adam, SGD, ...)
    :optim              : The given optimizer.
    :lr                 : Learning rate.
    :lr_step_size       : Steps takes to the next learning rate decay.
    :lr_decay_gamma     : The gamma-value used to dacay the learning rate.
    :train_loss         : List that records the average train loss for each batch
                         (of the train dataloader) for each epoch.
    :train_acc          : List that records the accuracy on the train dataset for
                         each epoch.
    :test_loss          : List that records the average test loss for each batch
                         (of the test dataloader) for each epoch.
    :test_acc           : List that records the accuracy on the test dataset for
                         each epoch.
    :model_path         : The path used to save the best models.

    '''
    def __init__(self,
            epoches: int, 
            train_dataloader: DataLoader, 
            test_dataloader: DataLoader, 
            model: nn.Module, 
            loss_fn: str = "CrossEntropyLoss", 
            opt_name: str = "Adam",
            optim: torch.optim.Adam =None,
            lr: float = 0.001,
            lr_step_size: int = 5,
            lr_decay_gamma: float = 0.9):
        ''' Initiate the attributes. '''
        self.epoches = epoches
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.opt = opt_name
        self.optim = optim
        self.lr = lr
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        self.lr_step_size = lr_step_size
        self.lr_decay_gamma = lr_decay_gamma

        self.model_path = r'./models/'

    def train(self):
        '''
        Train the given model for epoches.
        '''
        device = next(self.model.parameters()).device

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        
        best_model_params = copy.deepcopy(self.model.state_dict())
        best_acc = 0


        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        if self.loss_fn == "CrossEntropyLoss":
            loss_fn = nn.CrossEntropyLoss()
        if self.loss_fn == "MSELoss":
            loss_fn = nn.MSELoss()

        if self.optim==None:

            if self.opt == "Adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            if self.opt == "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            optimizer = self.optim


        
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_decay_gamma
        )

        for epoch in range(self.epoches):
        
            #train
            self.model.train()
            train_size = len(self.train_dataloader.dataset)
            num_train_batches = len(self.train_dataloader)


            # train_loss_batches: Accumulate loss from all batches.
            # correct: Accumulate the correct samples.
            train_loss_batches, correct = 0, 0

            for x, y in self.train_dataloader:
                y = y.float()
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)  # logits
                loss = loss_fn(pred, y)
                # core code
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    train_loss_batches += loss.item()

            epoch_train_acc = correct / train_size
            epoch_train_loss = train_loss_batches / num_train_batches

            # test
            self.model.eval()
            test_size = len(self.test_dataloader.dataset)
            num_test_batches = len(self.test_dataloader)

            test_loss_batches, test_correct = 0, 0 

            with torch.no_grad():
                for x, y in self.test_dataloader:
                    y = y.float()
                    x = x.to(device)
                    y = y.to(device)
                    pred = self.model(x)
                    loss = loss_fn(pred, y)
                    test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    test_loss_batches += loss.item()

            epoch_test_acc = test_correct / test_size
            epoch_test_loss = test_loss_batches / num_test_batches

            if epoch_test_acc > best_acc:
                best_model_params = copy.deepcopy(self.model.state_dict())
                best_acc = epoch_test_acc

            self.train_acc.append(epoch_train_acc)
            self.train_loss.append(epoch_train_loss)
            self.test_acc.append(epoch_test_acc)
            self.test_loss.append(epoch_test_loss)


            template = ('epoch:{:2d}, '\
                        'train_loss:{:.5f}, '\
                        'train_acc:{:.2f}%, '\
                        'test_loss:{:.5f}, '\
                        'test_acc:{:.2f}%')
            print(template.format(epoch, 
                                  epoch_train_loss, 
                                  epoch_train_acc*100, 
                                  epoch_test_loss, 
                                  epoch_test_acc*100))
            
            # Learning Rate Decay
            exp_lr_scheduler.step()
        
        self.model.load_state_dict(best_model_params)
        torch.save(self.model.state_dict(), 
                   os.path.join(self.model_path,'best_model.pth'))
        print("Done")

    def plot(self):
        '''
        Plot:
        :train_loss         : List that records the average train loss for each batch
                         (of the train dataloader) for each epoch.
        :train_acc          : List that records the accuracy on the train dataset for
                         each epoch.
        :test_loss          : List that records the average test loss for each batch
                         (of the test dataloader) for each epoch.
        :test_acc           : List that records the accuracy on the test dataset for
                         each epoch.
        '''
        plt.figure(figsize=(15,5),dpi=300)
        plt.subplot(1,2,1)
        plt.plot(range(self.epoches), self.train_loss, label='train_loss')
        plt.plot(range(self.epoches), self.test_loss, label='test_loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(range(self.epoches), self.train_acc, label='train_acc')
        plt.plot(range(self.epoches), self.test_acc, label='test_acc')
        plt.legend()
        plt.show()

class LocalizationFit():
    '''
    Train the given model.

    Attributes:
        epoches             : The epoches trained.
        train_dataloader    : The train dataloader.
        test_dataloader     : The test dataloader
        model               : The model whose paranmeters are trained.
        loss_fn             : The chosen loss function. 
                            (nn.CrossEntropyLoss, ...)
        optimizer           : The optimizer used to train the model.
        lr                  : Learning rate.
        lr_step_size        : Steps takes to the next learning rate decay.
        lr_decay_gamma      : The gamma-value used to dacay the learning rate.
        train_loss          : List that records the average train loss for each
                            batch (of the train dataloader) for each epoch.
        test_loss           : List that records the average test loss for each 
                            batch (of the test dataloader) for each epoch.
        exp_lr_scheduler    : The sheduler used to decay the larning rate.
        model_path          : The path used to save the best models.

    Methods:
        __init__()          : Initiate the attributes.
        _fit()              : Training process of one epoch.
        train()             : Train the model for the given epoches.
        plot()              : Plot the loss during the training process.

    '''
    def __init__(self,
            epoches: int, 
            train_dataloader: DataLoader, 
            test_dataloader: DataLoader, 
            model: nn.Module, 
            loss_fn: str = "MSELoss", 
            opt_name: str = "Adam",
            optim: torch.optim.Adam =None,
            lr: float = 0.001,
            lr_step_size: int = 7,
            lr_decay_gamma: float = 0.1):
        ''' Initiate the attributes. '''
        self.epoches = epoches
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model

        # loss function
        if loss_fn == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss()
        if loss_fn == "MSELoss":
            self.loss_fn = nn.MSELoss()


        self.lr = lr
        # optimizer
        if optim==None:
            if opt_name == "Adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                                  lr=self.lr)
            if opt_name == "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                                 lr=self.lr)
        else:
            self.optimizer = optim

        
        self.train_loss = []
        self.test_loss = []


        self.exp_lr_scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_decay_gamma
        )
        self.model_path = r'./models/'

    def _fit(self,
             epoch, 
             trainloader, 
             testloader):
        '''
        One epoch.
        '''
        total = 0
        running_loss = 0
        
        device = next(self.model.parameters()).device

        self.model.train()
        for x, y1, y2, y3, y4 in trainloader:
            x, y1, y2, y3, y4 = (x.to(device), 
                                 y1.to(device), y2.to(device),
                                 y3.to(device), y4.to(device))       
            y_pred1, y_pred2, y_pred3, y_pred4 = self.model(x)
            
            loss1 = self.loss_fn(y_pred1, y1)
            loss2 = self.loss_fn(y_pred2, y2)
            loss3 = self.loss_fn(y_pred3, y3)
            loss4 = self.loss_fn(y_pred4, y4)
            loss = loss1 + loss2 + loss3 + loss4
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
        # learning rate decay
        self.exp_lr_scheduler.step()
        epoch_loss = running_loss / len(trainloader)
            
            
        test_total = 0
        test_running_loss = 0 
        
        self.model.eval()
        with torch.no_grad():
            for x, y1, y2, y3, y4 in testloader:
                x, y1, y2, y3, y4 = (x.to(device), 
                                     y1.to(device), y2.to(device),
                                     y3.to(device), y4.to(device))
                y_pred1, y_pred2, y_pred3, y_pred4 = self.model(x)
                loss1 = self.loss_fn(y_pred1, y1)
                loss2 = self.loss_fn(y_pred2, y2)
                loss3 = self.loss_fn(y_pred3, y3)
                loss4 = self.loss_fn(y_pred4, y4)
                loss = loss1 + loss2 + loss3 + loss4
                test_running_loss += loss.item()
                
        epoch_test_loss = test_running_loss / len(testloader)
        
            
        print(
            'epoch: ', epoch, 
            'loss: ', round(epoch_loss, 3),
            'test_loss: ', round(epoch_test_loss, 3),
        )
            
        return epoch_loss, epoch_test_loss
    
    def train(self):
        '''
        Train the given model for epoches.
        '''
        device = next(self.model.parameters()).device

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        
        best_model_params = copy.deepcopy(self.model.state_dict())
        min_loss = 1e9


        # Train & test.
        for epoch in range(self.epoches):
            epoch_loss, epoch_test_loss = self._fit(
                epoch,
                self.train_dataloader,
                self.test_dataloader
            )

            if epoch_test_loss < min_loss:
                best_model_params = copy.deepcopy(self.model.state_dict())
                min_loss = epoch_test_loss

            self.train_loss.append(epoch_loss)
            self.test_loss.append(epoch_test_loss)
            
            
        
        self.model.load_state_dict(best_model_params)
        torch.save(self.model.state_dict(), 
                   os.path.join(self.model_path,'best_model.pth'))
        print("Done")

    def plot(self):
        '''
        Plot:
            train_loss          : List that records the average train loss for each
                                batch (of the train dataloader) for each epoch.
            test_loss           : List that records the average test loss for each 
                                batch (of the test dataloader) for each epoch.
        '''
        plt.figure(figsize=(15,5),dpi=300)
        plt.plot(range(self.epoches), self.train_loss, label='train_loss')
        plt.plot(range(self.epoches), self.test_loss, label='test_loss')
        plt.legend()
        plt.show()

class SemSegFit():
    '''
    Train the given model.

    Attributes:
        epoches             : The epoches trained.
        train_dataloader    : The train dataloader.
        test_dataloader     : The test dataloader
        model               : The model whose paranmeters are trained.
        loss_fn             : The chosen loss function. 
                            (nn.CrossEntropyLoss, ...)
        optimizer           : The optimizer used to train the model.
        lr                  : Learning rate.
        lr_step_size        : Steps takes to the next learning rate decay.
        lr_decay_gamma      : The gamma-value used to dacay the learning rate.
        train_loss          : List that records the average train loss for each
                            batch (of the train dataloader) for each epoch.
        test_loss           : List that records the average test loss for each 
                            batch (of the test dataloader) for each epoch.
        exp_lr_scheduler    : The sheduler used to decay the larning rate.
        model_path          : The path used to save the best models.

    Methods:
        __init__()          : Initiate the attributes.
        _fit()              : Training process of one epoch.
        train()             : Train the model for the given epoches.
        plot()              : Plot the loss during the training process.

    '''
    def __init__(self,
            epoches: int, 
            train_dataloader: DataLoader, 
            test_dataloader: DataLoader, 
            model: nn.Module, 
            loss_fn: str = "CrossEntropyLoss", 
            opt_name: str = "Adam",
            optim: torch.optim.Adam =None,
            lr: float = 0.001,
            lr_step_size: int = 7,
            lr_decay_gamma: float = 0.1):
        ''' Initiate the attributes. '''
        self.epoches = epoches
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model

        # loss function
        if loss_fn == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss()
        if loss_fn == "MSELoss":
            self.loss_fn = nn.MSELoss()


        self.lr = lr
        # optimizer
        if optim==None:
            if opt_name == "Adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                                  lr=self.lr)
            if opt_name == "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                                 lr=self.lr)
        else:
            self.optimizer = optim

        
        self.train_loss = []
        self.test_loss = []


        self.exp_lr_scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_decay_gamma
        )
        self.model_path = r'./models/'

    def train(self):
        '''
        Train the given model for epoches.
        '''
        device = next(self.model.parameters()).device

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        
        best_model_params = copy.deepcopy(self.model.state_dict())
        best_acc = 0


        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        for epoch in range(self.epoches):
            epoch_train_iou = []
            #train
            self.model.train()
            train_size = len(self.train_dataloader.dataset)
            num_train_batches = len(self.train_dataloader)


            # train_loss_batches: Accumulate loss from all batches.
            # correct: Accumulate the correct samples.
            train_loss_batches, correct, total = 0, 0, 0

            for x, y in self.train_dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)  # logits
                loss = self.loss_fn(pred, y)
                # core code
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    y_pred = torch.argmax(pred, dim=1)
                    correct += (y_pred == y).sum().item()
                    total += y.size(0)
                    train_loss_batches += loss.item()

                    intersection = torch.logical_and(y, y_pred)
                    union = torch.logical_or(y, y_pred)
                    batch_iou = torch.true_divide(
                        torch.sum(intersection), 
                        torch.sum(union)
                    )
                    epoch_train_iou.append(batch_iou)

            epoch_train_acc = correct / (total*256*256)
            epoch_train_loss = train_loss_batches / num_train_batches

            # test
            epoch_test_iou = []
            self.model.eval()
            test_size = len(self.test_dataloader.dataset)
            num_test_batches = len(self.test_dataloader)

            test_loss_batches, test_correct, test_total = 0, 0, 0 

            with torch.no_grad():
                for x, y in self.test_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                    y_pred = torch.argmax(pred, dim=1)
                    test_correct += (y_pred == y).sum().item()
                    test_total += y.size(0)
                    test_loss_batches += loss.item()

                    intersection = torch.logical_and(y, y_pred)
                    union = torch.logical_or(y, y_pred)
                    batch_iou = torch.true_divide(
                        torch.sum(intersection), 
                        torch.sum(union)
                    )
                    epoch_test_iou.append(batch_iou)

            epoch_test_acc = test_correct / (test_total*256*256)
            epoch_test_loss = test_loss_batches / num_test_batches

            if epoch_test_acc > best_acc:
                best_model_params = copy.deepcopy(self.model.state_dict())
                best_acc = epoch_test_acc

            self.train_acc.append(epoch_train_acc)
            self.train_loss.append(epoch_train_loss)
            self.test_acc.append(epoch_test_acc)
            self.test_loss.append(epoch_test_loss)


            template = (
                'epoch:{:2d}, '\
                'train_loss:{:.5f}, '\
                'train_acc:{:.2f}%, '\
                'train_IoU:{:.3f}, '\
                'test_loss:{:.5f}, '\
                'test_acc:{:.2f}% '\
                'test_IoU:{:.3f}. '
            )

            print(template.format(epoch, 
                                  epoch_train_loss, 
                                  epoch_train_acc*100, 
                                  torch.mean(torch.Tensor(epoch_train_iou)),
                                  epoch_test_loss, 
                                  epoch_test_acc*100,
                                  torch.mean(torch.Tensor(epoch_test_iou))))
            
            # Learning Rate Decay
            self.exp_lr_scheduler.step()
        
        self.model.load_state_dict(best_model_params)
        torch.save(self.model.state_dict(), 
                   os.path.join(self.model_path,'best_model.pth'))
        print("Done")

    def plot(self):
        '''
        Plot:
        :train_loss         : List that records the average train loss for each batch
                         (of the train dataloader) for each epoch.
        :train_acc          : List that records the accuracy on the train dataset for
                         each epoch.
        :test_loss          : List that records the average test loss for each batch
                         (of the test dataloader) for each epoch.
        :test_acc           : List that records the accuracy on the test dataset for
                         each epoch.
        '''
        plt.figure(figsize=(15,5),dpi=300)
        plt.subplot(1,2,1)
        plt.plot(range(self.epoches), self.train_loss, label='train_loss')
        plt.plot(range(self.epoches), self.test_loss, label='test_loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(range(self.epoches), self.train_acc, label='train_acc')
        plt.plot(range(self.epoches), self.test_acc, label='test_acc')
        plt.legend()
        plt.show()