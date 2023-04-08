
# coding=utf-8
from __future__ import print_function
import sys
import os

import importlib, argparse

from comm.utils import PlatENASConfig

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("-gpu_id", "--gpu_id", help="GPU ID", type=str)
    _parser.add_argument("-file_id", "--file_id", help="file id", type=str)
    # _parser.add_argument("-uuid", "--uuid", help="uuid of the individual", type=str)

    # _parser.add_argument("-super_node_ip", "--super_node_ip", help="ip of the super node", type=str)
    # _parser.add_argument("-super_node_pid", "--super_node_pid", help="pid on the super node", type=int)
    # _parser.add_argument("-worker_node_ip", "--worker_node_ip", help="ip of this worker node", type=str)

    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu_id
    from datetime import datetime
    import multiprocessing
    from algs.performance_pred.utils import TrainConfig
    from ast import literal_eval
    from comm.registry import Registry
    from train.utils import OptimizerConfig, LRConfig

    from compute.redis import RedisLog
    from compute.pid_manager import PIDManager
    import torch
    import traceback
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.backends.cudnn as cudnn




class TrainModel(object):
    def __init__(self, file_id, logger): #file_id -> fileUri

        # module_name = 'scripts.%s'%(file_name)
        module_name = file_id
        if module_name in sys.modules.keys():
            self.log_record('Module:%s has been loaded, delete it' % (module_name))
            del sys.modules[module_name]
            _module = importlib.import_module('.', module_name)
        else:
            _module = importlib.import_module('.', module_name)

        net = _module.EvoCNNModel()

        # def init_weights(m):
        #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
        #         nn.init.xavier_uniform_(m.weight)
        # net.apply(init_weights)

        g = PlatENASConfig('algorithm')
        algs_name = g.read_ini_file('run_algorithm')
        work_name = g.read_ini_file('name')

        self.algs_name = algs_name
        self.work_name = work_name

        # 需要将~改成..
        self.ptSavedPath = os.path.join(os.path.expanduser('~'), "onebinary/ENASLogs/", algs_name ,"checkpoints/", work_name)
        self.bestPtSavedUri = os.path.join(self.ptSavedPath, file_id + ".pt")

        # self.ptSavedPath = os.path.join("./")
        if not os.path.exists(self.bestPtSavedUri):
            self.ptSavedPath = os.path.join('./')
            self.bestPtSavedUri = os.path.join(self.ptSavedPath, file_id + ".pt")
            if not os.path.exists(self.bestPtSavedUri):
                print("dont have the pt file.")
                exit(0)

        checkpoint = torch.load(self.bestPtSavedUri)
        net.load_state_dict(checkpoint['model'])



        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_error = 1.0
        best_loss = 10
        
        # 加载best_loss
        try:
            best_loss = checkpoint['loss']
        except:
            best_loss = 10
        
        best_acc = 0.0
        self.net = net

        TrainConfig.ConfigTrainModel(self)
        # initialize optimizer
        o = OptimizerConfig()
        opt_cls = Registry.OptimizerRegistry.query(o.read_ini_file('_name'))
        opt_params = {k: (v) for k, v in o.read_ini_file_all().items() if not k.startswith('_')}
        l = LRConfig()
        lr_cls = Registry.LRRegistry.query(l.read_ini_file('lr_strategy'))
        lr_params = {k: (v) for k, v in l.read_ini_file_all().items() if not k.startswith('_')}
        lr_params['lr'] = float(lr_params['lr'])
        opt_params['lr'] = float(lr_params['lr'])
        self.opt_params = opt_params
        self.opt_cls = opt_cls
        self.opt_params['total_epoch'] = self.nepochs
        self.lr_params = lr_params
        self.lr_cls = lr_cls
        # after the initialization

        self.criterion = criterion
        self.best_error = best_error
        self.best_loss = best_loss
        self.best_acc = best_acc

        self.best_error_test = best_error
        self.best_loss_test = best_loss

        self.file_id = file_id
        self.logger = logger
        self.counter = 0


        self.last_epoch_time = 0

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))
        print('[%s]-%s' % (dt, _str))

    def get_optimizer(self, epoch):
        # get optimizer
        self.opt_params['current_epoch'] = epoch
        opt_cls_ins = self.opt_cls(**self.opt_params)
        optimizer = opt_cls_ins.get_optimizer(filter(lambda p: p.requires_grad, self.net.parameters()))
        return optimizer

    def get_learning_rate(self, epoch):
        self.lr_params['optimizer'] = self.get_optimizer(epoch)
        self.lr_params['current_epoch'] = epoch
        lr_cls_ins = self.lr_cls(**self.lr_params)
        learning_rate = lr_cls_ins.get_learning_rate()
        return learning_rate

    def train(self, epoch):
        # 如果loss不衰减了, 那么削减学习率
        if self.counter / 8 == 1:
            self.counter = 0
            self.lr = self.lr * 0.5
        # if epoch / 8 == 1:
        #     self.lr = self.lr * 0.5
        self.net.train()

        # optimizer = self.get_optimizer(epoch)
        import torch.optim as optim
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
        self.log_record('Train-Epoch:%3d,  Loss: %.5f, Acc:%.5f' % (epoch + 1, running_loss / total, (correct / total)))

    # 最终验证 valid
    def main_val(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0

        self.last_epoch_time = int(datetime.now().timestamp())
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
        if 1 - correct / total < self.best_error:
            self.best_error = 1 - correct / total
            torch.save({'model': self.net.state_dict(), 'acc': 1 - self.best_error}, os.path.join(self.ptSavedPath, self.file_id + ".v2.fullepoch.pt"))

        if test_loss / total < self.best_loss:
            self.log_record('Validation loss decreased ({%.6f} --> {%.6f}).  Saving model ...'%(self.best_loss, test_loss / total))
            self.best_loss = test_loss / total

            # 保存loss最佳的模型
            # torch.save({'model': self.net.state_dict(), 'loss': self.best_loss}, os.path.join(self.ptSavedPath, self.file_id + ".fullepoch.pt"))

            self.counter = 0
        else:
            self.counter += 1
        
        if correct / total > self.best_acc:
            self.best_acc = correct / total

        self.log_record('Valid-Epoch:%3d, Loss:%.5f, Acc:%.5f, lr:%.5f' % (epoch+1, test_loss / total, correct / total, self.lr))

        now = int(datetime.now().timestamp())
        if now - self.last_epoch_time >= 60 * 5: #最少1分钟保存一次
            """
            以时间为单位保存模型.
            """
            # torch.save({'model': self.net.state_dict(), 'loss': self.best_loss, 'epoch': epoch}, os.path.join(self.ptSavedPath, self.file_id + ".epochtmp.pt"))
            _tmp_uri = os.path.join("./", self.file_id + ".%d.epoch%d.pt" % (now, self.nepoch + epoch))
            torch.save({'model': self.net.state_dict(), 'loss': self.best_loss, 'epoch': epoch}, _tmp_uri)
            print("interval save tmp pt to [%s]" % _tmp_uri)
            pass

    def main_test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
        if 1 - correct / total < self.best_error_test:
            self.best_error_test = 1 - correct / total
            self.best_loss_test = test_loss / total

        self.log_record('Test-Epoch:%3d, Loss:%.5f, Acc:%.5f, lr:%.5f, now_best_acc:%.5f, best_acc_loss:%.5f' % (epoch+1, test_loss / total, correct / total, self.lr, 1-self.best_error_test, self.best_loss_test))


    def process(self):
        total_epoch = self.nepochs
        scheduler = self.get_learning_rate(total_epoch)

        full_epoch = self.fullepochs

        num_para = sum([p.numel() for p in self.net.parameters()])
        self.log_record('file_id:%s, parameters:%d' % (self.file_id, num_para))
        for p in range(full_epoch):
            self.train(p)
            self.main_val(p)
            self.main_test(p)
            scheduler.step()
        return self.best_acc, self.best_error, self.best_loss, num_para # 0.1系数模拟一下实验


class RunModel(object):
    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def do_work(self, gpu_id, file_id, uuid = ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        logger = RedisLog("full-v3-" + os.path.basename(file_id) + '.txt')
        best_error = 100.0
        best_loss = 99999999999999999999.0
        num_para = 2100000000
        best_acc = 0.0

        best_test_acc = 0.0

        try:
            m = TrainModel(file_id, logger)

            m.log_record(
                '[%s, %s, %s] Used GPU#%s, worker name:%s[%d]' % (os.path.basename(__file__), m.algs_name, m.work_name, gpu_id, multiprocessing.current_process().name, os.getpid()))
            best_acc, best_error, best_loss, num_para = m.process()
        except BaseException as e:
            msg = traceback.format_exc()
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            print('%s' % (msg))
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Exception occurs:%s' % (msg)
            logger.info('[%s]-%s' % (dt, _str))
        finally:
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Finished-Error:%.5f, Best-Loss:%.5f, num-para:%d, Finished-Acc:%.5f' % (best_error, best_loss, num_para, best_acc)
            logger.info('[%s]-%s' % (dt, _str))

            #logger.write_file('RESULTS', 'results.txt', '%s=%.5f,%.5f,%d\n' % (file_id, best_error, best_loss, num_para))
            #_str = '%s;%.5f;%.5f;%d\n' % (uuid, best_error, best_loss, num_para)
            # logger.write_file('CACHE', 'cache.txt', _str)


if __name__ == "__main__":
    # 加入redis监视. 当Ctrl+C需要关闭时, 自行退出
    # PIDManager.WorkerEnd.add_worker_pid(_args.super_node_ip, _args.super_node_pid, _args.worker_node_ip)

    # 还需要写一个在空闲时候, 自动执行further训练的模块

    # 执行
    RunModel().do_work(_args.gpu_id, _args.file_id)

    # PIDManager.WorkerEnd.remove_worker_pid(_args.super_node_ip, _args.super_node_pid, _args.worker_node_ip)