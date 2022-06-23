from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch

class L1(nn.Module):
    def __init__(self, args, features, num_classes,logger):
        super(L1, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        # print(self.features)
        # print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        return x


def make_layers(cfg, args, logger ):
    layers = []
    in_channels = 4
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            if args.mode == "WAGE":
                conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
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



cfg_list = {
    'l1': [('C', 64, 4, 'same', 2.0)]
}

def l1( args, logger, pretrained=None):
    cfg = cfg_list['l1']
    layers = make_layers(cfg, args, logger)
    model = L1(args, layers, num_classes=10, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

