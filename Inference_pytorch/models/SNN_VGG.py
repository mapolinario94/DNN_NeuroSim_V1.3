from utee import misc
print = misc.logger.info
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch


class SNN_VGG(nn.Module):
    def __init__(self, args, logger):
        super(SNN_VGG, self).__init__()
        # assert isinstance(features, nn.Sequential), type(features)
        self.conv1 = QConv2d(3, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv'+str(1)+'_', model=args.model)

        self.conv2 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(2) + '_', model=args.model)

        self.max_pool_1 = nn.MaxPool2d(2, 2)

        self.conv3 = QConv2d(64, 128, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(3) + '_', model=args.model)

        self.conv4 = QConv2d(128, 128, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(4) + '_', model=args.model)

        self.max_pool_2 = nn.MaxPool2d(2, 2)

        self.conv5 = QConv2d(128, 256, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(5) + '_', model=args.model)

        self.conv6 = QConv2d(256, 256, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(6) + '_', model=args.model)

        self.conv7 = QConv2d(256, 256, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(7) + '_', model=args.model)

        self.max_pool_3 = nn.MaxPool2d(2, 2)

        self.conv8 = QConv2d(256, 512, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(8) + '_', model=args.model)

        self.conv9 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(9) + '_', model=args.model)

        self.conv10 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(10) + '_', model=args.model)

        self.max_pool_4 = nn.MaxPool2d(2, 2)

        self.conv11 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(11) + '_', model=args.model)

        self.conv12 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(12) + '_', model=args.model)

        self.conv13 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(13) + '_', model=args.model)




        self.classifier1 = QLinear(in_features=2048, out_features=4096,
                                  logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                  wl_error=args.wl_error,
                                  wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                                  cellBit=args.cellBit,
                                  subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t,
                                  v=args.v, detect=args.detect, target=args.target,
                                  name='FC' + str(1) + '_', model=args.model)

        self.classifier2 = QLinear(in_features=4096, out_features=4096,
                                   logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                   wl_error=args.wl_error,
                                   wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                                   cellBit=args.cellBit,
                                   subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t,
                                   v=args.v, detect=args.detect, target=args.target,
                                   name='FC' + str(2) + '_', model=args.model)

        self.classifier3 = QLinear(in_features=4096, out_features=100,
                                   logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                   wl_error=args.wl_error,
                                   wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                                   cellBit=args.cellBit,
                                   subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t,
                                   v=args.v, detect=args.detect, target=args.target,
                                   name='FC' + str(3) + '_', model=args.model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool_1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool_2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.max_pool_3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = x.view(x.size(0), -1)

        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)

        return x


def snn_vgg(args, logger, pretrained=None):
    model = SNN_VGG(args, logger=logger)
    return model