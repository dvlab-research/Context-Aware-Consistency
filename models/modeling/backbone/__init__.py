from models.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, pretrained=True):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained=pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
