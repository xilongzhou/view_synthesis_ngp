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
        # model = network.MetaInterpolation(args)
        # self.writer.add_graph(trainer.model.model, (torch.Tensor(1, 6, 256, 448).cuda(), ))

    def tensorboard_prepare_summary(self, data):
        summary = {}
        # {'frameNeighbor':neighbors_imgs, 'frameGen':torch.clamp(generated, min=0.0, max=1.0), 'frameTrgt':reference_img}

        summary['Input_Gen'] = torchvision.utils.make_grid(
            [data['frameNeighbor'].cpu()[0],
             data['frameTrgt'].cpu()[0]], padding=10)
        
        summary['Gen_GT'] = torchvision.utils.make_grid(
            [data['frameTest'].cpu()[0],
             data['frameTrgt'].cpu()[0]], padding=10)
        
        transf = transforms.ToTensor()
        F_r = transf(flowToColor(data['flow'][0].permute(1,2,0).cpu().detach().numpy()))
        summary['Flow'] = torchvision.utils.make_grid(
            [F_r], padding=10
        )

        if 'outer_lr' in data.keys():
            summary['outer_lr'] = data['outer_lr']
        
        return summary

    def tensorboard_log_summary(self, train_summary, val_summary, itr):
        # Scalars
        self.writer.add_scalars('Loss/train_L1', 
                                {'train_L1': train_summary['L1']}, itr)
        self.writer.add_scalars('Loss/validation_L1', 
                                {'validation_L1': val_summary['loss'], 'validation_L1_innerstep0': val_summary['loss_is0']}, itr)

        self.writer.add_scalars('PSNR', {'psnr': val_summary['psnr'], 'psnr_innerstep0': val_summary['psnr_is0']}, itr)

        self.writer.add_scalars('Learning Rate', {'outer_lr': train_summary['outer_lr']}, itr)
        
        # Images
        self.writer.add_image('Validation Output', val_summary['Gen_GT'], itr)
        self.writer.add_image('Validation Input', val_summary['Input_Gen'], itr)
        self.writer.add_image('Validation Flow', val_summary['Flow'], itr)
        
        self.writer.add_image('Train Output', train_summary['Gen_GT'], itr)
        self.writer.add_image('Train Input', train_summary['Input_Gen'], itr)
        self.writer.add_image('Train Flow', train_summary['Flow'], itr)

        # print update
        print(" Iterations: %4d  TrainL1: %0.6f  TrainExecTime: %0.1f  ValL1:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f" \
            % (itr, train_summary['L1'], train_summary['exec_time'], val_summary['loss'], val_summary['psnr'], val_summary['exec_time']))

        return