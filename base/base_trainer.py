import os, json, math, logging, sys, datetime, time
import torch
from torch.utils import tensorboard
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.helpers import dir_exists

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:
    def __init__(self, model, resume, config, iters_per_epoch, train_logger=None, gpu=None, test=False):
        self.model = model
        self.config = config

        if gpu == 0:
            self.train_logger = train_logger
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
            log_dir = os.path.join(config['trainer']['log_dir'], config['experim_name'])
            log_path = os.path.join(log_dir, '{}.log'.format(time.time()))
            dir_exists(log_dir)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.info("config: {}".format(self.config))
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False
        self.gpu = gpu 
        torch.cuda.set_device(self.gpu)

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        trainable_params = [{'params': list(filter(lambda p:p.requires_grad, self.model.get_other_params()))},
                            {'params': list(filter(lambda p:p.requires_grad, self.model.get_backbone_params())), 
                            'lr': config['optimizer']['args']['lr'] / 10}]

        self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(), device_ids=[gpu], find_unused_parameters=True)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params) # trainable_params should be obtained before wraping the model with DistributedDataParallel
        
        model_params = sum([i.shape.numel() for i in list(filter(lambda p: p.requires_grad, model.parameters()))])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])

        assert opt_params == model_params, 'some params are missing in the opt'

        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler'])(optimizer=self.optimizer, num_epochs=self.epochs, 
                                        iters_per_epoch=iters_per_epoch)

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        if self.gpu == 0:
            # CHECKPOINTS & TENSOBOARD
            date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
            run_name = config['experim_name']
            self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], run_name)
            helpers.dir_exists(self.checkpoint_dir)
            config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
            with open(config_save_path, 'w') as handle:
                json.dump(self.config, handle, indent=4, sort_keys=True)
            
            writer_dir = os.path.join(cfg_trainer['log_dir'], run_name)
            self.writer = tensorboard.SummaryWriter(writer_dir)

        self.test = test
        if resume: self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus



    def train(self):
        if self.test:
            results = self._valid_epoch(0)
            if self.gpu == 0:
                self.logger.info('\n')
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')
            return 

        for epoch in range(self.start_epoch, self.epochs+1):
            self._train_epoch(epoch)
            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                if self.gpu == 0:
                    self.logger.info('\n\n Epoch {}:'.format(epoch))
                    for k, v in results.items():
                        self.logger.info(f'         {str(k):15s}: {v}')
                
                log = {'epoch' : epoch, **results}
                if self.gpu == 0:
                    if self.train_logger is not None:
                        self.train_logger.add_entry(log)

                # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
                if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                    try:
                        if self.mnt_mode == 'min': self.improved = (log[self.mnt_metric] < self.mnt_best)
                        else: self.improved = (log[self.mnt_metric] > self.mnt_best)
                    except KeyError:
                        self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                        break
                        
                    if self.improved:
                        self.mnt_best = log[self.mnt_metric]
                        self.not_improved_count = 0
                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count > self.early_stoping:
                        # if (not self.dist) or (self.dist and self.gpu == 0):
                        if self.gpu == 0:
                            self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                            self.logger.warning('Training Stoped')
                        break

                if self.gpu == 0:
                    # SAVE CHECKPOINT
                    self._save_checkpoint(epoch, save_best=self.improved)
        if self.gpu == 0:
            self.logger.info(str(self.train_logger))


    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        filename = os.path.join(self.checkpoint_dir, f'checkpoint.pth')
        
        self.logger.info(f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth {} at {} epoch".format(self.mnt_best, epoch))

    def _resume_checkpoint(self, resume_path):
        
        if self.gpu == 0:
            self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f'Error when loading: {e}')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if self.gpu == 0:
            if "logger" in checkpoint.keys():
                self.train_logger = checkpoint['logger']
            self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
