from utee import misc
print = misc.logger.info
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch

class L1(nn.Module):
    def __init__(self, args, features, num_classes, out_ch, logger):
        super(L1, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        # print(self.features)
        # print(self.classifier)
        self.classifier = make_layers([('L', out_ch, num_classes)], args, logger, in_ch=0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class SFN(nn.Module):
    def __init__(self, args, logger):
        super(SFN, self).__init__()
        # assert isinstance(features, nn.Sequential), type(features)
        self.conv1 = QConv2d(4, 64, kernel_size=3, padding=1, stride=2,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv'+str(1)+'_', model=args.model)
        self.conv2 = QConv2d(64, 128, kernel_size=3, padding=1, stride=2,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(2) + '_', model=args.model)
        self.conv3 = QConv2d(128, 256, kernel_size=3, padding=1, stride=2,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(3) + '_', model=args.model)
        self.conv4 = QConv2d(256, 512, kernel_size=3, padding=1, stride=2,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(4) + '_', model=args.model)

        self.conv_r11 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                                logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                                onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                                ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                                target=args.target, name='Conv' + str(5) + '_', model=args.model)

        self.conv_r12 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                                logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                                onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                                ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                                target=args.target, name='Conv' + str(6) + '_', model=args.model)

        self.conv_r21 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                                logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                                onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                                ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                                target=args.target, name='Conv' + str(7) + '_', model=args.model)

        self.conv_r22 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                                logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                                onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                                ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                                target=args.target, name='Conv' + str(8) + '_', model=args.model)

        self.deconv3 = QConv2d(512, 128, kernel_size=4, padding=2, stride=1,
                               logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                               wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                               onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                               ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                               target=args.target, name='Conv' + str(9) + '_', model=args.model)

        self.deconv2 = QConv2d(416, 64, kernel_size=4, padding=2, stride=1,
                               logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                               wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                               onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                               ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                               target=args.target, name='Conv' + str(10) + '_', model=args.model)

        self.deconv1 = QConv2d(224, 4, kernel_size=4, padding=2, stride=1,
                               logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                               wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                               onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                               ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                               target=args.target, name='Conv' + str(11) + '_', model=args.model)

        self.up43 = QConv2d(512, 32, kernel_size=4, padding=2, stride=1,
                            logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                            wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                            onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                            ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                            target=args.target, name='Conv' + str(12) + '_', model=args.model)

        self.up32 = QConv2d(416, 32, kernel_size=4, padding=2, stride=1,
                            logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                            wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                            onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                            ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                            target=args.target, name='Conv' + str(13) + '_', model=args.model)

        self.up21 = QConv2d(224, 32, kernel_size=4, padding=2, stride=1,
                            logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                            wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                            onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                            ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                            target=args.target, name='Conv' + str(14) + '_', model=args.model)

        self.up10 = QConv2d(100, 32, kernel_size=4, padding=2, stride=1,
                            logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                            wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                            onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                            ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                            target=args.target, name='Conv' + str(15) + '_', model=args.model)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = QLinear(in_features=32, out_features=1,
                                  logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                                  wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                  subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                  name='FC'+str(1)+'_', model = args.model)

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11)  # + out_conv4
        out_rconv12 = (1 - out_rconv12) * out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21)  # + out_rconv12
        out_rconv22 = (1 - out_rconv22) * out_rconv12

        out_rconv22 = F.pad(out_rconv22, [8, 8, 8, 8])

        up43 = self.up43(out_rconv22)
        flow4_up = crop_like(up43, out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        concat3 = F.pad(concat3, [16, 16, 16, 16])
        up32 = self.up32(concat3)
        flow3_up = crop_like(up32, out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        concat2 = F.pad(concat2, [32, 32, 32, 32])
        up21 = self.up21(concat2)
        flow2_up = crop_like(up21, out_conv1)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        concat1 = F.pad(concat1, [64, 64, 64, 64])
        up10 = self.up10(concat1)

        x = self.max_pool(up10)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, args, logger, in_ch ):
    layers = []
    in_channels = in_ch
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            padding = v[3]
            # if v[3] == 'same':
            #     padding = v[2]//2
            # else:
            #     padding = 0
            if args.mode == "WAGE":
                conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding, stride=v[4],
                                 logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                                 wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(i)+'_', model = args.model)
            elif args.mode == "FP":
                conv2d = FConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(i)+'_' )
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d]
            in_channels = out_channels
        if v[0] == 'L':
            if args.mode == "WAGE":
                linear = QLinear(in_features=v[1], out_features=v[2],
                                logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                                wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                name='FC'+str(i)+'_', model = args.model)
            elif args.mode == "FP":
                linear = FLinear(in_features=v[1], out_features=v[2],
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name='FC'+str(i)+'_')
            if i < len(cfg)-1:
                non_linearity_activation =  nn.ReLU()
                layers += [linear, non_linearity_activation]
            else:
                layers += [linear]
    return nn.Sequential(*layers)


# ('C', out_channles,  kernel_size, padding, stride)
cfg_list = {
    'l1': [('C', 64,  3, 1, 2), ('M', 128, 128)],
    'l2': [('C', 128, 3, 1, 2), ('M', 64, 64)],
    'l3': [('C', 256, 3, 1, 2), ('M', 32, 32)],
    'l4': [('C', 512, 3, 1, 2), ('M', 16, 16)],
    'l5': [('C', 512, 3, 1, 1), ('M', 16, 16)],
    'l6': [('C', 512, 3, 1, 1), ('M', 16, 16)],
    'l7': [('C', 512, 3, 1, 1), ('M', 16, 16)],
    'l8': [('C', 512, 3, 1, 1), ('M', 16, 16)],
    'l9': [('C', 128, 4, 0, 1), ('M', 32, 32)],
    'l10': [('C', 64, 4, 0, 1), ('M', 64, 64)],
    'l11': [('C', 4,  4, 0, 1), ('M', 128, 128)],
    'l12': [('C', 32, 4, 0, 1), ('M', 32, 32)],
    'l13': [('C', 32, 4, 0, 1), ('M', 64, 64)],
    'l14': [('C', 32, 4, 0, 1), ('M', 128, 128)],
    'l15': [('C', 32, 4, 0, 1), ('M', 256, 256)],
    'l16': [('C', 64, 3, 1, 1), ('M', 256, 256)],
    'l17': [('C', 64, 3, 1, 1), ('M', 256, 256)],
    'l18': [('C', 64, 3, 1, 2), ('C', 128, 3, 1, 2),
            ('C', 256, 3, 1, 2), ('C', 512, 3, 1, 2),
            ('C', 512, 3, 1, 1), ('C', 512, 3, 1, 1),
            ('C', 512, 3, 1, 1), ('C', 512, 3, 1, 1), ('M', 16, 16)],
}

cfg_list_in_ch = {
    'l1': 4,
    'l2': 64,
    'l3': 128,
    'l4': 256,
    'l5': 512,
    'l6': 512,
    'l7': 512,
    'l8': 512,
    'l9': 512,
    'l10': 416,
    'l11': 224,
    'l12': 512,
    'l13': 416,
    'l14': 224,
    'l15': 100,
    'l16': 32,
    'l17': 4,
    'l18': 4,
}


def l1( args, logger, pretrained=None):
    cfg = cfg_list[args.model]
    layers = make_layers(cfg, args, logger, cfg_list_in_ch[args.model])
    model = L1(args, layers, num_classes=1, out_ch=cfg[-2][1], logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def sfn(args, logger, pretrained=None):
    model = SFN(args, logger=logger)
    return model



