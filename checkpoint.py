'''

Save function : Weight, Loss, Plot save  !

'''

import os
import torch
import datetime
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from option import args


class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True
        self.dir = opt.save
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')


    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)

    def done(self):
        self.log_file.close()


    def set_epoch(self, ep):
        self.epoch = ep
    
    def save_results_nopostfix(self, filename, sr, scale):
        if not args.test_only:
            apath = '{}/results/{}_train/x{}/epoch{}'.format(self.dir, self.opt.data_test, scale,self.epoch)
        else:
            apath = '{}/results/{}/x{}'.format(self.dir, self.opt.data_test, scale)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        normalized = sr[0].data.mul(255 / self.opt.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        if not args.test_only:
            imageio.imwrite('{}_epoch{}.jpg'.format(filename, self.epoch), ndarr)
        else:
            imageio.imwrite('{}.jpg'.format(filename), ndarr)