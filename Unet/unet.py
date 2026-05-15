import segmentation_models_pytorch as smp


def get_model():
    model = smp.Unet(
    encoder_name="resnet34", # feature extractor backbone (goes down)
    encoder_weights="imagenet",   # pretrained on ImageNet, transfer learning
    in_channels=3, # rgb channels on tensor
    classes=1, # output has 1 channel
    activation=None, # BCEWithLogitsLoss applies sigmoid internally
    )
    
    print(f"[Model] Parameters: {count_parameters(model):,}")
    
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # Returns total number of trainable parameters in the model.
