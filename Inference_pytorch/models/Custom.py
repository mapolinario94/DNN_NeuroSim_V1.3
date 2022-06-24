from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch

class L1(nn.Module):
    def __init__(self, args, features, num_classes,out_ch, logger):
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
    'l3': [('C', 128, 3, 1, 2), ('M', 64, 64)],
    'l4': [('C', 128, 3, 1, 2), ('M', 16, 16)],
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
}

cfg_list_in_ch = {
    'l1': 4,
    'l2': 64,
    'l3': 64,
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
    'l15': 100
}

def l1( args, logger, pretrained=None):
    cfg = cfg_list[args.model]
    layers = make_layers(cfg, args, logger, cfg_list_in_ch[args.model])
    model = L1(args, layers, num_classes=1, out_ch=cfg[0][1],logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model



