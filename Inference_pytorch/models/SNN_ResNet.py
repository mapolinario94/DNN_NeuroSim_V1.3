from utee import misc
print = misc.logger.info
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch


class SNN_ResNet(nn.Module):
    def __init__(self, args, logger):
        super(SNN_ResNet, self).__init__()
        # assert isinstance(features, nn.Sequential), type(features)
        self.conv1 = QConv2d(3, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv'+str(1)+'_', model=args.model)

        self.max_pool_1 = nn.MaxPool2d(2, 2)

        self.conv2 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(2) + '_', model=args.model)

        self.conv3 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(3) + '_', model=args.model)

        self.conv4 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(4) + '_', model=args.model)

        self.conv5 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(5) + '_', model=args.model)

        self.conv6 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(6) + '_', model=args.model)

        self.conv7 = QConv2d(64, 64, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(7) + '_', model=args.model)

        #Stage2

        self.conv8 = QConv2d(64, 128, kernel_size=3, padding=1, stride=2,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(8) + '_', model=args.model)

        self.conv9 = QConv2d(128, 128, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(9) + '_', model=args.model)

        self.conv10 = QConv2d(64, 128, kernel_size=1, padding=0, stride=2,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(10) + '_', model=args.model)

        self.conv11 = QConv2d(128, 128, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(11) + '_', model=args.model)

        self.conv12 = QConv2d(128, 128, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(12) + '_', model=args.model)

        # Stage3

        self.conv13 = QConv2d(128, 256, kernel_size=3, padding=1, stride=2,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(13) + '_', model=args.model)

        self.conv14 = QConv2d(256, 256, kernel_size=3, padding=1, stride=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                             target=args.target, name='Conv' + str(14) + '_', model=args.model)

        self.conv15 = QConv2d(128, 256, kernel_size=1, padding=0, stride=2,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(15) + '_', model=args.model)

        self.conv16 = QConv2d(256, 256, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(16) + '_', model=args.model)

        self.conv17 = QConv2d(256, 256, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(17) + '_', model=args.model)

        # Stage4

        self.conv18 = QConv2d(256, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(18) + '_', model=args.model)

        self.conv19 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(19) + '_', model=args.model)

        self.conv20 = QConv2d(256, 512, kernel_size=1, padding=0, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(20) + '_', model=args.model)

        self.conv21 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(21) + '_', model=args.model)

        self.conv22 = QConv2d(512, 512, kernel_size=3, padding=1, stride=1,
                              logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                              wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                              onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                              ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect,
                              target=args.target, name='Conv' + str(22) + '_', model=args.model)

        self.max_pool_2 = nn.MaxPool2d(2, 2)

        self.classifier = QLinear(in_features=2048, out_features=100,
                                  logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                                  wl_error=args.wl_error,
                                  wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                                  cellBit=args.cellBit,
                                  subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t,
                                  v=args.v, detect=args.detect, target=args.target,
                                  name='FC' + str(1) + '_', model=args.model)



    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool_1(x)

        #S1

        x1 = self.conv2(x)
        x1 = self.conv3(x1)
        x = (1-x1)*x

        x1 = self.conv4(x)
        x1 = self.conv5(x1)
        x = (1 - x1) * x

        x1 = self.conv6(x)
        x1 = self.conv7(x1)
        x = (1 - x1) * x

        #S2
        x1 = self.conv8(x)
        x1 = self.conv9(x1)
        x = (1 - x1) * self.conv10(x)

        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x = (1 - x1) * x

        #S3
        x1 = self.conv13(x)
        x1 = self.conv14(x1)
        x = (1 - x1) * self.conv15(x)

        x1 = self.conv16(x)
        x1 = self.conv17(x1)
        x = (1 - x1) * x

        # S4
        x1 = self.conv18(x)
        x1 = self.conv19(x1)
        x = (1 - x1) * self.conv20(x)

        x1 = self.conv21(x)
        x1 = self.conv22(x1)
        x = (1 - x1) * x

        x = self.max_pool_2(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


def snn_resnet(args, logger, pretrained=None):
    model = SNN_ResNet(args, logger=logger)
    return model