from utee import misc
print = misc.logger.info
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch


class FireFlowNet(nn.Module):
    def __init__(self, args, logger):
        super(FireFlowNet, self).__init__()
        # assert isinstance(features, nn.Sequential), type(features)
        self.conv1 = QConv2d(4, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv'+str(1)+'_', model=args.model)

        self.conv2 = QConv2d(32, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(2) + '_', model=args.model)

        self.conv3 = QConv2d(32, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(3) + '_', model=args.model)

        self.conv4 = QConv2d(32, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(4) + '_', model=args.model)

        self.conv5 = QConv2d(32, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(5) + '_', model=args.model)

        self.conv6 = QConv2d(32, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(6) + '_', model=args.model)

        self.conv7 = QConv2d(32, 32, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(7) + '_', model=args.model)

        self.max_pool = nn.MaxPool2d(256, 256)

        self.classifier = QLinear(in_features=32, out_features=1,
                                  logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                  wl_error=args.wl_error,
                                  wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                                  cellBit=args.cellBit,
                                  subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t,
                                  v=args.v, detect=args.detect, target=args.target,
                                  name='FC' + str(1) + '_', model=args.model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.max_pool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


def fireflownet(args, logger, pretrained=None):
    model = FireFlowNet(args, logger=logger)
    return model
