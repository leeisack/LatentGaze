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

        latent =  latent_vector.view(latent_vector.shape[0], -1)[:,:3584]
        # print(latent.size())

        feature = torch.cat((feature, latent), 1)
        # print(feature.size())
        output = self.fc(feature)

        return output
    
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

class GazeEstimationModelResnet101(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelResnet101, self).__init__()
        feature_extractor = models.resnet101(pretrained=True)

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
        
        self.reduction = nn.Linear(2048, 300)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True


        self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=300, out_features=2)
        GazeEstimationAbstractModel._init_weights(self.modules())

class GazeEstimationModelVGG16(GazeEstimationAbstractModel):
    def __init__(self, num_out=2):
        super(GazeEstimationModelVGG16, self).__init__()
        feature_extractor = models.vgg16(pretrained=True)

        feature_extractor_modules = [module for module in feature_extractor.features]
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        feature_extractor_modules.append(self.AdaptiveAvgPool2d)
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)

        # self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.reduction = nn.Linear(512, 300)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=300, out_features=2)
        GazeEstimationAbstractModel._init_weights(self.modules())

class GazeEstimationModelLatent(GazeEstimationAbstractModelLatent):
    def __init__(self, num_out=2):
        super(GazeEstimationModelLatent, self).__init__()
        self.fc = GazeEstimationAbstractModelLatent._create_fc_layers(in_features=1024, out_features=2)
        GazeEstimationAbstractModelLatent._init_weights(self.modules())


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

        self.fc = GazeEstimationAbstractModel_img_with_latent._create_fc_layers(in_features=4096, out_features=2)
        GazeEstimationAbstractModel_img_with_latent._init_weights(self.modules())

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

       
def make_model(data_train, model):
    if data_train == "MPII_train":
        if model == "ResNet101":
            return GazeEstimationModelResnet101()
        elif model == "VGG16":
            return GazeEstimationModelVGG16()
    if model == "Img_with_Latent":
        print("Img_with_Latent")
        return GazeEstimationModel_img_with_Latent()

    elif data_train == "Latent_train":
        return GazeEstimationModelLatent()

class GazeModel(nn.Module):
    def __init__(self, opt):
        super(GazeModel, self).__init__()
        print('\nMaking Gaze model...')
        self.opt = opt
        self.n_GPUs = opt.n_GPUs
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.model = make_model(opt.data_train, opt.model).to(self.device)
        self.load(opt.pre_train, cpu=opt.cpu)

    def load(self, pre_train, cpu=False):
        
        #### load gaze model ####
        if pre_train != '.':
            print('Loading gaze model from {}'.format(pre_train))
            self.model.load_state_dict(
                torch.load(pre_train),
                strict=True
            )
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