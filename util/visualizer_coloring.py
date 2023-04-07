import torch
import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from computeFlowColor import computeImg as flowToColor

class Visualizer():
    # Tensorboard Summary and terminal log
    def __init__(self, args, trainer):
        # Initialize tensorboard log writer
        self.writer = SummaryWriter('logs/' + args.experiment)
        # self.writer.add_graph(trainer.model.model, (torch.Tensor(1, 6, 256, 448).cuda(), ))

    def tensorboard_prepare_summary(self, data):
        summary = {}

        summary['Input'] = torchvision.utils.make_grid(
            [data['frameTrgt'].cpu()[0],
             data['frame_gray'].expand(-1, 3, -1, -1).cpu()[0]], padding=10)
        
        summary['Output'] = torchvision.utils.make_grid(
            [data['frameTest'].cpu()[0]], padding=10)
        
        return summary

    def tensorboard_log_summary(self, train_summary, val_summary, itr):
        # Scalars
        self.writer.add_scalars('Loss/train_L1', 
                                {'train_L1': train_summary['L1']}, itr)
        self.writer.add_scalars('Loss/validation_L1', 
                                {'validation_L1': val_summary['loss'], 'validation_L1_innerstep0': val_summary['loss_is0']}, itr)

        self.writer.add_scalars('PSNR', {'psnr': val_summary['psnr'], 'psnr_innerstep0': val_summary['psnr_is0']}, itr)
        
        # Images
        self.writer.add_image('Validation Output', val_summary['Output'], itr)
        self.writer.add_image('Validation GT and Input', val_summary['Input'], itr)
        
        self.writer.add_image('Train Output', train_summary['Output'], itr)
        self.writer.add_image('Train GT and Input', train_summary['Input'], itr)

        # print update
        print(" Iterations: %4d  TrainL1: %0.6f  TrainExecTime: %0.1f  ValL1:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f" \
            % (itr, train_summary['L1'], train_summary['exec_time'], val_summary['loss'], val_summary['psnr'], val_summary['exec_time']))

        return