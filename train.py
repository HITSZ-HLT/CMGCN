import logging
import argparse
import math
import os
import sys
import random
import numpy
from sklearn import metrics
from time import strftime, localtime
from pytorch_pretrained import BertModel
import torch
import torch.nn as nn
from data_utils import  Tokenizer4Bert, Dataset
from models import CMGCN
from bucket_iterator import BucketIterator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.trainset = Dataset(opt.dataset_file['train'], tokenizer)
        self.valset = Dataset(opt.dataset_file['val'], tokenizer)
        self.testset = Dataset(opt.dataset_file['test'], tokenizer)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader,test_data_loader,max_test_acc,max_test_f1):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = self.opt.path
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/{0}_{1}'.format(self.opt.model_name, self.opt.dataset)
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            
            for i_batch, batch in enumerate(train_data_loader):
                self.model.train()
                global_step += 1
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device)   for col in self.opt.inputs_cols if col!='text']
                outputs = self.model(inputs)
                targets = batch['label'].to(self.opt.device)

                loss = criterion(outputs, targets) 
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                if global_step % self.opt.log_step == 0:
                    val_acc, val_f1,val_precision,val_recall = self._evaluate_acc_f1(val_data_loader)
                    logger.info('> max_val_f1: {:.4f}, max_val_acc: {:.4f},  max_test_f1: {:.4f}, max_test_acc: {:.4f}'.format(
                        max_val_f1,max_val_acc,max_test_f1,max_test_acc))
                    logger.info('> val_acc: {:.4f}, val_f1: {:.4f},val_precision: {:.4f},val_recall: {:.4f}'.format(val_acc, val_f1,val_precision,val_recall))

                    if val_acc > max_val_acc:
                        max_val_f1 = val_f1
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        
                        torch.save(self.model.state_dict(), path)
                        logger.info('>> saved: {}'.format(path))


            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        self.model.load_state_dict(torch.load(path))
        test_acc, test_f1,test_precision,test_recall = self._evaluate_acc_f1(test_data_loader,macro = True)
        test_acc_, test_f1_,test_precision_,test_recall_ = self._evaluate_acc_f1(test_data_loader)
        return (test_acc, test_f1,test_precision,test_recall) , (test_acc_, test_f1_,test_precision_,test_recall_)

    def _evaluate_acc_f1(self, data_loader,macro=False,pre = None):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device)    for col in self.opt.inputs_cols if col!='text']
                t_targets = t_batch['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = torch.argmax(t_outputs_all, -1).cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall

    def run(self):
        criterion = nn.CrossEntropyLoss()
        self._reset_params()
        optimizer = self.opt.optimizer([{'params':self.model.bert.parameters(),'lr':2e-5}],lr=self.opt.lr, weight_decay=self.opt.l2reg)
        train_data_loader = BucketIterator(data=self.trainset.data, batch_size=self.opt.batch_size, shuffle=True,opt = self.opt)
        test_data_loader = BucketIterator(data=self.testset.data, batch_size=self.opt.batch_size, shuffle=False,opt = self.opt)
        val_data_loader = BucketIterator(data=self.valset.data, batch_size=self.opt.batch_size, shuffle=False,opt = self.opt)
        
        if self.opt.pre != "":
            bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
            model_ = self.opt.model_class(bert, self.opt).to(self.opt.device)
            model_.load_state_dict(torch.load(self.opt.model_path))
            res = self._evaluate_acc_f1(test_data_loader,pre = self.opt.pre)
            print(res)
            return
        (test_acc, test_f1 ,test_precision , test_recall),(test_acc_, test_f1_ ,test_precision_ , test_recall_)= self._train(criterion, optimizer,train_data_loader,val_data_loader,test_data_loader,0,0)
        print("{} {} {} {}\n".format(test_acc, test_f1 ,test_precision , test_recall) + "{} {} {} {}".format(test_acc_, test_f1_ ,test_precision_ , test_recall_))
        return "{} {} {} {}\n".format(test_acc, test_f1 ,test_precision , test_recall) + "{} {} {} {}".format(test_acc_, test_f1_ ,test_precision_ , test_recall_)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='CMGCN', type=str)
    parser.add_argument('--dataset', default='box_top10', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=20, type=int)
    parser.add_argument('--pretrained_bert_name', default='./bert_base_uncased', type=str)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default="cuda:1", type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=777, type=int, help='set seed for reproducibility')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--path', default="", type=str)
    parser.add_argument('--pre', default="",type=str)
    parser.add_argument('--model_path', default="",type=str)
    parser.add_argument('--macro_f1', default=False,type=bool)
    opt = parser.parse_args()

    model_classes = {
        'CMGCN':CMGCN,
    }
    dataset_files = {
        'box_top10':
        {
            'train': 'processed_train.data',
            'test': 'processed_test.data',
            'val':'processed_valid.data'
        },
    }
    input_colses = {
        'CMGCN':['bert_indices', 'graph','box_vit'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    seed = opt.seed 
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    ins = Instructor(opt)
    log = ins.run()
    print(log)

if __name__ == '__main__':
    main()
