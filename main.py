from multiprocessing.spawn import freeze_support
from rt_gene.gaze_estimation_models_pytorch import GazeModel
import os
import utility
import data
import loss  
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
os.environ['KMP_DUPLICATE_LIB_OK']='True'


utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    gaze_model = GazeModel(args).cuda()
    
    loss = loss.Loss(args, checkpoint)
    t = Trainer(args, loader, gaze_model, loss, checkpoint)

    def main():
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()

    if __name__ == '__main__':  
        freeze_support()  
        main()




