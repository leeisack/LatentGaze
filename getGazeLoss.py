import torch
from functools import partial
import numpy as np


_loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum")
        }
_param_num = {
    "mse": 2
}

def computeGazeLoss(angular_out, gaze_batch_label):
    _criterion = _loss_fn.get("mse")()
    gaze_loss = _criterion(angular_out, gaze_batch_label).cuda()
    pi_angular_error = 0
    theta_angular_error = 0
    detected_number = angular_out.shape[0]
    for i in range(detected_number):

        # COMPUTE PI ANGULAR ERROR
        EST_pi = angular_out[i][0].cpu().detach().numpy()
        GT_pi = gaze_batch_label[i][0].cpu().detach().numpy()


        if EST_pi > GT_pi:
            pi_angular_error += np.abs(EST_pi - GT_pi)
        else:
            pi_angular_error += np.abs(GT_pi - EST_pi)
    
         # COMPUTE THETA ANGULAR ERROR
        EST_theta = angular_out[i][1].cpu().detach().numpy()
        GT_theta = gaze_batch_label[i][1].cpu().detach().numpy()

        if EST_theta > GT_theta:
            theta_angular_error += np.abs(EST_theta - GT_theta)
        else:
            theta_angular_error += np.abs(GT_theta - EST_theta)

    angular_error = (pi_angular_error + theta_angular_error)*45
        
    return gaze_loss, angular_error
