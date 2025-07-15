import os
import os.path as osp
#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import math
import time
import argparse
import logging 
from safetensors.torch import load_file
import yaml
import cProfile
from tqdm import tqdm
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import numpy as np  

#import torch._dynamo 
#from pysteps.visualization import plot_precip_field
import wandb 
import torch.distributed as dist
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
if __name__ == "__main__":
    if is_main_process():
        wandb.init(
        project="Diffcast",
        
        #notes='training a larger vqvae with lpips with 18M params, with 512 and 1024 n_embed and codebook size'
        )
    else:
        wandb.require("service")
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from dataset.get_datasets import get_dataset
from utils.metrics import Evaluator
from utils.tools import print_log, cycle, show_img_info

# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "03fe58695b6f176f52334de3e701e0a00ccd5a80"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"

def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',       type=str,   default='phydnet',           help='backbone model for deterministic prediction')
    parser.add_argument('--stochastic',     type=str,   default='diffusion',           help='diffusion model for stochastic prediction')
    parser.add_argument('--use_diff',       action="store_true", default=True,        help='Weather use diff framework, as for ablation study')
    parser.add_argument('--backbone_config',     type=str,   default=None,           help='path to backbone config file, needed only for BlockGPT')
    
    parser.add_argument("--seed",           type=int,   default=0,              help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='Outputs/',   help="experiment directory")
    

    parser.add_argument("--debug",          type=bool,  default=False,           help="load a small dataset for debugging")
    parser.add_argument("--profiler",       type=bool,  default=False,           help="use profiler to check the code")


    # --------------- Dataset ---------------
    parser.add_argument("--dataset",        type=str,   default='sevir',        help="dataset name")
    parser.add_argument("--img_size",       type=int,   default=128,            help="image size")
    parser.add_argument("--img_channel",    type=int,   default=1,              help="channel of image")
    parser.add_argument("--seq_len",        type=int,   default=9,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",      type=int,   default=3,              help="number of frames to input")
    parser.add_argument("--frames_out",     type=int,   default=6,             help="number of frames to output")    
    parser.add_argument("--temp_res_sevir",   type=int,   default=5,              help="SEVIR ONLY: Time resolusion")
    parser.add_argument("--seq_len_sevir",   type=int,   default=None,              help="SEVIR ONLY: To select larger sequence and then downsample")
    parser.add_argument("--num_workers",    type=int,   default=4,              help="number of workers for data loader")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=0.7e-5,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=1000,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=6,              help="batch size")
    parser.add_argument("--epochs",         type=int,   default=30,              help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=200000,          help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default="Exps/basic_exps/Singlephydnet/sevir/2024-09-10_13-07-49_None/checkpoints/ckpt-49621.pt",            help="resumed checkpoint milestone")
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    parser.add_argument("--testing",        type=bool,  default=False,              help="testing saved model")
    parser.add_argument("--traininig",      type=bool,  default=True,                 help="training model")
    
    parser.add_argument("--wandb_state",    type=str,   default='disabled',      help="wandb state config")
    parser.add_argument("--config",         type=str,   default=None,            help="config file path")
    parser.add_argument("--buidl_dirs",     type=bool,  default=True,            help="build dirs for experiment")

    args = parser.parse_args()
    return args


class Runner(object):
    
    def __init__(self, args):
        
        self.args = args
        
        # overwrite args with config file if exists
        if self.args.config is not None:
            with open(str(self.args.config), "r") as f:
                config = yaml.safe_load(f)
            self.update_args(self.args, config)
            
        set_seed(self.args.seed)
        
        self._preparation()
        
        if self.args.buidl_dirs:
            self._build_dirs()
        
        if self.args.debug:
            self.args.batch_size = 1
        
        # Config DDP kwargs from accelerate
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        
        self.accelerator = Accelerator(
            project_config  =   project_config,
            kwargs_handlers =   [ddp_kwargs, process_kwargs],
            mixed_precision =   self.args.mixed_precision,
            log_with        =   "wandb",
        )
        
        # Config log tracker 'wandb' from accelerate
        # self.accelerator.init_trackers(
        #     project_name=f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}",
        #     config=self.args.__dict__,
        #     init_kwargs={"wandb": 
        #         {
        #         "mode": self.args.wandb_state,
        #         # 'resume': self.args.ckpt_milestone
        #         }
        #                  }   # disabled, online, offline
        # )
        
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self._build_model()
        self._build_optimizer()
        
        # distributed ema for parallel sampling

        self.model, self.optimizer,  self.scheduler, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.scheduler,
            self.train_loader, 
            self.valid_loader, self.test_loader
        )
        
        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            # print_log(show_img_info(sample), self.is_main)
            
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def update_args(self, args, config_dict):
        for key, val in config_dict.items():
            setattr(args, key, val) 
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.model_name = ('Diff' if self.args.use_diff else 'Single') + self.args.backbone
        
        if not self.args.testing:

            self.date_time  = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name   = f"{self.date_time}_{self.model_name}_{self.args.dataset}"
            
           # cur_dir         = os.path.dirname(os.path.abspath(__file__))
            self.exp_dir    = osp.join(self.args.exp_dir,self.exp_name)        
            self.ckpt_path  = osp.join(self.exp_dir, 'checkpoints')
            self.valid_path = osp.join(self.exp_dir, 'valid_samples')
            self.test_path  = osp.join(self.exp_dir, 'test_samples')
            self.log_path   = osp.join(self.exp_dir, 'logs')
            self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
            
        else:
            assert self.args.ckpt_milestone is not None, "Please provide a milestone for testing"
            #Exps/basic_exps/Singlephydnet/sevir/2024-09-10_13-07-49_None/checkpoints/ckpt-49621.pt
            self.exp_dir = '/'.join(self.args.ckpt_milestone.split('/')[:-2])
            self.ckpt_path = osp.join(self.exp_dir, 'checkpoints')
            self.valid_path = osp.join(self.exp_dir, 'valid_samples')
            self.test_path = osp.join(self.exp_dir, 'test_samples')
            self.log_path = osp.join(self.exp_dir, 'logs')
            self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
            
        
    def _build_dirs(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
                # logging.StreamHandler()
            ]
        )
        
    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================
        if self.args.seq_len_sevir is None:
            self.args.seq_len_sevir = self.args.seq_len
        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,#128
            seq_len=self.args.seq_len_sevir,#25
            temp_res_sevir = self.args.temp_res_sevir,
            batch_size=self.args.batch_size,#6
            debug=self.args.debug
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
        if self.args.dataset != 'sevir' and self.args.dataset != 'knmi':
            # preload big batch data for gradient accumulation
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size*self.args.grad_acc_step, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
            
        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}",
                  self.is_main)
        
    def _build_model(self):
        # =================================
        # import and create different models given model config
        # =================================

        if self.args.backbone == 'simvp':
            from models.simvp import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        
        elif self.args.backbone == 'phydnet':
            from models.phydnet import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "device": self.device
            }
            model = get_model(**kwargs)
        elif self.args.backbone == 'blockGPTBackbone':
            from models.blockGPTBackbone import get_model
            from models.blockGPTBackbone import BlockGPTBackboneConfig
            with open(self.args.backbone_config) as f:
                kwargs = json.load(f)
                config = BlockGPTBackboneConfig(**kwargs)
            model = get_model(config)        
        else:
            raise NotImplementedError
        #turng off grad for backbone 
        for param in model.parameters():
             param.requires_grad = False
        
        if self.args.use_diff:
            print("diff is being used")
            from models.diffcast import get_model
            freeze_backbone = True if self.args.backbone == 'blockGPTBackbone' else False
            kwargs = {
                'img_channels' : self.args.img_channel,
                'dim' : 64,
                'dim_mults' : (1,2,4,8),
                'T_in': self.args.frames_in,
                'T_out': self.args.frames_out,
                'sampling_timesteps': 250,
                'freeze_backbone': freeze_backbone
            }
            diff_model = get_model(**kwargs)
            diff_model.load_backbone(model)
            model = diff_model
        
        
        self.model = model
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)


    def _build_optimizer(self):
        # =================================
        # Calcutate training nums and config optimizer and learning schedule
        # =================================
        num_steps_per_epoch = len(self.train_loader)
        num_epoch = math.ceil(self.args.training_steps / num_steps_per_epoch) if not self.args.debug else -math.inf #num_epoch gets equal to 80000 when debugging because of the small dataset considered
        
        self.global_epochs = max(num_epoch, self.args.epochs)
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps = self.args.warmup_steps

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.global_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps , 
                num_training_steps=self.global_steps,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    self.args.scheduler
            )
        )
            
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"optimizer: {self.optimizer} with init lr: {self.args.lr}")
        
    
    def save(self):
        # =================================
        # Save checkpoint state for model and ema
        # =================================
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),

        }
        
        torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
        print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        
        
    def load(self, milestone):
        # =================================
        # load model checkpoint
        # =================================        
        device = self.accelerator.device
        
        if '.pt' in str(milestone):
            data = torch.load(milestone, map_location=device)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.model = self.accelerator.prepare(model)
        
        self.optimizer.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        
        if self.is_main:
            self.ema.load_state_dict(data['ema'])

        self.cur_epoch = data['epoch']
        self.cur_step = data['step']
        print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
        
    def train_backbone(self):
        pbar = tqdm(
            initial=self.cur_step,
            total=self.global_steps,
            disable=not self.is_main,
        )
        start_epoch = self.cur_epoch
        
        for epoch in range(start_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                epoch_loss = 0.0
                num_batches = 0
                # train the model with mixed_precision
                with torch.autocast("cuda", dtype=torch.float16): 

                    radar_batch = self._get_seq_data(batch)
                    frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
                    assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
                    loss, _ = self.model(input_tensor=frames_in, target_tensor=frames_out)
                    epoch_loss+=loss
                    num_batches+=1                    
                    self.accelerator.backward(loss)
                    #print("loss", loss) 
                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)   
                wandb.log({"loss": loss.item(), "global step": self.cur_step})
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                
                # record train info
                wandb.log({"loss item":loss.item()})
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr
                log_dict['loss'] = loss.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)   
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
                pbar.set_description(state_str)
                
                # update ema param and log file every 20 steps
                if i % 20 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
                pbar.update(1)
                
                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon= self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            if self.is_main:
                                for i in range(radar_ori.shape[0]):
                                    self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(self.sanity_path, f"{i}/vil"),data_type='vil')
                                    
                            break

                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)
                
            # save checkpoint and do test every 5 epoch
            if (epoch+1) % 1 == 0:
                self.save()

            print_log(f" ========= Finisth one Epoch ==========", self.is_main)
            avg_loss = epoch_loss/num_batches
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
        self.accelerator.end_training()
        wandb.finish()        

        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch      # [B, T, C, H, W]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        loss, backbone_loss, diff_loss = self.model.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        return {'total_loss': loss, "backbone_loss": backbone_loss, "diff_loss": diff_loss}
        
    
    @torch.no_grad()
    def _sample_batch(self, batch):
        
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        radar_pred = self.model.inference(radar_input) if self.accelerator.num_processes ==1 else self.model.module.inference(radar_input)
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()

        return radar_gt, radar_pred
    
    
    def test_samples(self, milestone, do_test=False):
        # init test data loader
        data_loader = self.test_loader if do_test else self.valid_loader
        # init sampling method
        self.model.eval()
        # init test dir config
        cnt = 0
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        print("#####################", self.is_main)
        if self.is_main:
            eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )
        # start test loop
        for batch in tqdm(data_loader,desc='Test Samples', disable=not self.is_main):
            # sample
            radar_ori, radar_recon= self._sample_batch(batch)
            # evaluate result and save
            eval.evaluate(radar_ori, radar_recon)
            if self.is_main:
                for i in range(radar_ori.shape[0]):
                    self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(save_dir, f"{cnt}-{i}/vil"),data_type='vil')

            self.accelerator.wait_for_everyone()
            # cnt += 1
            # if cnt > 10:
            #     break
        # test done
        if self.is_main:
            res = eval.done()
            print_log(f"Test Results: {res}")
            print_log("="*30)

        
    def check_milestones(self, target_ckpt=None):

        mils_paths = os.listdir(self.ckpt_path)
        milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)
        
        """
        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
        """
        
        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], do_test=True)
            
def main():
    args = create_parser()
    exp = Runner(args)
    if args.traininig:
        exp.train_backbone()
        #exp.train()
    if args.testing:
        assert args.buidl_dirs == False, "Save test samples in same directories of training"
        exp.check_milestones(target_ckpt=args.ckpt_milestone)
    

if __name__ == '__main__':
    # 测试代码各模块执行效率
    # pip install graphviz
    # pip install gprof2dot
    # gprof2dot -f pstats train.profile | dot -Tpng -o result.png
    # cProfile.run('main()', filename='train.profile', sort='cumulative')

    main() 