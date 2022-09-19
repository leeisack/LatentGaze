import torch
import torch.nn as nn
from torchvision import models


class GazeEstimationAbstractModel(nn.Module):
    
    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):

        output_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

        return  output_layer

    def forward(self, face_img):

        feature = self.feature_extractor(face_img)
        feature = feature.view(feature.shape[0], -1)
        reduction = self.reduction(feature)
        output = self.fc(reduction)

        return output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)


class GazeEstimationAbstractModelLatent(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModelLatent, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):

        output_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

        return  output_layer

    def forward(self, latent_vector):
        latent_vector = latent_vector.view(latent_vector.shape[0], -1)
        latent_vector = latent_vector[:, 1536:2560]
        # print(latent_vector.size())
        output = self.fc(latent_vector)

        return output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

class GazeEstimationAbstractModel_img_with_latent(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel_img_with_latent, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):

        output_layer = nn.Sequential(
            nn.Linear(in_features, out_features)
        )

        return  output_layer

    def forward(self, face_img, latent_vector):

        feature = self.feature_extractor(face_img)
        feature = feature.view(feature.shape[0], -1)
        
        b, c, h, w = latent_vector.size()
        latent_vector = latent_vector.squeeze(1)
        
        # latent, prob = self.CA_layer(latent_vector)
        latent =  latent_vector.view(latent_vector.shape[0], -1)

        feature = torch.cat((feature, latent), 1)
        # print(feature.size())
        output = self.fc(feature)
        return output, 0
    
    # class GazeEstimationAbstractModel_img_with_latent(nn.Module):
    
    # def __init__(self):
    #     super(GazeEstimationAbstractModel_img_with_latent, self).__init__()

    # @staticmethod
    # def _create_fc_layers(in_features, out_features):

    #     output_layer = nn.Sequential(
    #         nn.Linear(in_features, out_features)
    #     )

    #     return  output_layer

    # def forward(self, face_img, latent_vector):

    #     feature = self.feature_extractor(face_img)
    #     feature = feature.view(feature.shape[0], -1)
        
    #     b, c, h, w = latent_vector.size()
    #     latent_vector = latent_vector.squeeze(1)
        
    #     # latent, prob = self.CA_layer(latent_vector)

    #     latent =  latent_vector.view(latent_vector.shape[0], -1)

    #     feature = torch.cat((feature, latent), 1)
    #     print(feature.size())
    #     output = self.fc(feature)

    #     return output, prob

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

class GazeEstimationModel_img_with_Latent(GazeEstimationAbstractModel_img_with_latent):
    def __init__(self):
        super(GazeEstimationModel_img_with_Latent, self).__init__()
        feature_extractor = models.resnet18(pretrained=True)

        self.feature_extractor = nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.CA_layer = CALayer(channel=2, kernel_size=1)

        self.fc = GazeEstimationAbstractModel_img_with_latent._create_fc_layers(in_features=1536, out_features=2)
        GazeEstimationAbstractModel_img_with_latent._init_weights(self.modules())
       
def make_model(data_train, model):
    return GazeEstimationModel_img_with_Latent()

class CALayer(nn.Module):
    def __init__(self, channel, kernel_size=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # feature channel downscale and upscale --> channel weight
        self.adaptive_weight = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.adaptive_weight(y)
        return x * y, y
    
class GazeModel(nn.Module):
    def __init__(self, opt):
        super(GazeModel, self).__init__()
        print('\nMaking Gaze model...')
        self.opt = opt
        self.n_GPUs = opt.n_GPUs
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.model = make_model(opt.data_train, opt.model).to(self.device)
        self.load(opt.pre_train, cpu=opt.cpu)
        
    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load(self, pre_train, cpu=False):
        
        #### load gaze model ####
        if pre_train != '.':
            print('Loading gaze model from {}'.format(pre_train))
            pretrained_dict = torch.load(pre_train)
            pretrained_dict = self.remove_prefix(pretrained_dict, 'model.')
            self.check_keys(self.model, pretrained_dict)
            self.model.load_state_dict(pretrained_dict, strict=True)
            print("Complete loading Gaze estimation model weight")
        
        num_parameter = self.count_parameters(self.model)

        print(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M \n")

    def forward(self, imgs, latents):
        return self.model(imgs, latents)
    # def forward(self, imgs):
        # return self.model(imgs)

    def count_parameters(self, model):
        param_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return param_sum