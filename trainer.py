import torch
import numpy as np


import utility
from decimal import Decimal
from tqdm import tqdm
from option import args

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from getGazeLoss import *


def angular(gaze, label):
    assert gaze.size == 3, "The size of gaze must be 3"
    assert label.size == 3, "The size of label must be 3"

    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def gazeto3d(gaze):
    assert gaze.size == 2, "The size of gaze must be 2"
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

matplotlib.use("Agg")
class Trainer():
    def __init__(self, opt, loader, gaze_model, loss,ckp):
        self.opt = opt
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = gaze_model
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.gaze_model_scheduler = utility.make_gaze_model_scheduler(opt, self.optimizer)
        self.loss = loss
        self.error_last = 1e8
        self.iteration_count = 0
        self.endPoint_flag = True

    
    def train(self):
        total_gaze_loss = 0
        total_angular_error = 0
        total_prob = 0

        epoch = self.gaze_model_scheduler.last_epoch + 1
        lr = self.gaze_model_scheduler.get_last_lr()[0]
        self.ckp.set_epoch(epoch)

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        """
        TRAIN
        """
        
        self.iteration_count = 0
        for batch, (samples) in enumerate(self.loader_train):
            self.iteration_count += 1

            if self.iteration_count % args.test_every == 0:
                break
            

            imgs = samples["image"].type(torch.FloatTensor)
            imgs = imgs.to(torch.device("cuda"))
            labels_theta = samples['label'][0]
            labels_pi = samples['label'][1]
         
            labels = torch.stack([labels_theta, labels_pi], dim=1)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(torch.device("cuda"))
            
            latent = samples["latent"].type(torch.FloatTensor)
            latent = latent.to(torch.device("cuda"))

            self.optimizer.zero_grad() 
            
            timer_data.hold()
            timer_model.tic()
            

            select_idx = [3,4]
            latent = latent[:,:, select_idx,:]
            # angular_out, prob = self.model(imgs)
            angular_out, prob = self.model(imgs, latent)
            
            gaze_loss, angular_error = computeGazeLoss(angular_out, labels)

            total_prob += prob
            total_gaze_loss += gaze_loss
            total_angular_error += angular_error

            gaze_loss.backward()
            self.optimizer.step()
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t[Average Gaze Loss : {:.4f}]\t{:.1f}+{:.1f}s\t[Average Angular Error:{:.3f}]'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    total_gaze_loss / (self.opt.batch_size * (batch+1)),
                    timer_model.release(),
                    timer_data.release(),total_angular_error / (self.opt.batch_size * (batch+1))))
            timer_data.tic()

        # average_prob = total_prob.sum(dim=0) / (self.opt.batch_size)
        average_gaze_loss =  total_gaze_loss / (self.opt.batch_size * (batch+1))
        average_angular_error = total_angular_error / (self.opt.batch_size * (batch+1))

        # print('Train Channel Probability : ', average_prob)
        print('Train gaze loss : ', float(average_gaze_loss))
        print('Train Angular loss : ', float(average_angular_error))
        
        train_probability_path = "./experiment/Train_probability(%s).txt" %self.opt.model
        train_gaze_loss_path = "./experiment/Train_gaze_loss(%s).txt" %self.opt.model
        train_angular_error_path = "./experiment/Train_angular_loss(%s).txt" %self.opt.model
        path_list = [train_probability_path, train_gaze_loss_path, train_angular_error_path]
        log_list = [float(average_gaze_loss), float(average_angular_error)]

        for i in range(len(log_list)):
            txt = open(path_list[i], 'a')
            log = str(log_list[i]) + "\n"
            txt.write(log)
            txt.close()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
      
    def test(self):

        total_gaze_loss = 0
        total_angular_error = 0
        total_prob = 0
        epoch = self.gaze_model_scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()
        timer_test = utility.timer()


        with torch.no_grad():
            
            gaze_loss = 0
            angular_error = 0      
            ang_error = 0
            current = 0
            total_angular_error = 0
            total_angular_error_2 = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for _, (sample) in enumerate(tqdm_test):
                current += 1
                
                img = sample["image"].type(torch.FloatTensor)
                img = img.to(torch.device("cuda"))
                labels_theta = sample['label'][0]
                labels_pi = sample['label'][1]
                labels = torch.stack([labels_theta, labels_pi], dim=1)
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(torch.device("cuda"))
                
                latent = sample["latent"].to(torch.device("cuda"))

                # select_idx = [3, 4]
                select_idx = [4, 5]
                
                latent = latent[:,:, select_idx,:]
                angular_out, _ = self.model(img, latent)

                gaze_loss, ang_error = computeGazeLoss(angular_out, labels)      
                
                for k, gaze in enumerate(angular_out):
                    gaze = gaze.cpu().detach().numpy()
                    gt = labels.cpu().numpy()[k]

                    angular_error = angular(
                                gazeto3d(gaze),
                                gazeto3d(gt)
                            )
                total_angular_error += ang_error
                total_angular_error_2 += angular_error

            self.ckp.log[-1, 0] = total_gaze_loss / len(self.loader_test)
            avg_angular_error = total_angular_error / len(self.loader_test)
            avg_angular_error2 = total_angular_error_2 / len(self.loader_test)
            best = self.ckp.log.min(0)
            self.ckp.write_log(
                '[{}]\t Angular_error: {:.2f}, error2: {:.2f} (@epoch {})'.format(
                    self.opt.data_test,
                    avg_angular_error,
                    avg_angular_error2,
                    best[1][0] + 1
                )
            )

        if not self.opt.test_only:
            Gaze_model_save(self.opt, self.model, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

    def step(self):
        self.gaze_model_scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.gaze_model_scheduler.last_epoch
            return epoch >= self.opt.epochs

def Gaze_model_save(opt, gaze_model, epoch, is_best=False):
        path = opt.save
        name = 'gaze_model_latest_' + str(epoch) + ".pt"
        torch.save(
            gaze_model.state_dict(), 
            os.path.join(path, name)
        )
        if is_best:
                torch.save(
                    gaze_model.state_dict(),
                    os.path.join(path, 'gaze_model_best.pt')
                )