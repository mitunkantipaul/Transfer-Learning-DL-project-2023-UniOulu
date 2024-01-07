import timm
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def get_model(model_name, num_classes, pretrained=True, pretrained_path=None, dropout_rate=0.5):
    """
    Returns the specified model with the given number of output classes.

    Parameters:
    model_name (str): Name of the model (e.g., 'resnet18', 'vgg16', 'vit_base_patch16_224').
    num_classes (int): Number of classes for the final output layer.
    pretrained (bool): If True, returns a model pre-trained on ImageNet.
    dropout_rate (float): Dropout rate (used in some models).

    Returns:
    torch.nn.Module: Initialized PyTorch model.
    """

    if model_name == 'resnet18':
        if pretrained_path:
            # Load the pre-trained model from the specified path
            model = models.resnet18()
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict, strict=False)
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Create a new ResNet-18 model
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'resnet50':
        if pretrained_path:
            # Load the pre-trained model from the specified path
            model = models.resnet50(pretrained=pretrained)
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict, strict=False)
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Create a new ResNet-50 model
            model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'vgg16':
        if pretrained_path:
            # Load the pre-trained model from the specified path
            model = models.vgg16()
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict, strict=False)
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Create a new vgg16 model
            model = models.vgg16(pretrained=pretrained)

    elif model_name == 'vit':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # Add other models here as elif blocks

    else:
        raise ValueError("Invalid model name")

    # Example of adding dropout to a model
    if 'resnet' in model_name:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    
    return model


def get_optimizer(model, optimizer_name, lr, momentum=0.9, weight_decay=0.0, beta1=0.9, beta2=0.999):
    """
    Returns the specified optimizer with the given parameters.

    Parameters:
    model (torch.nn.Module): The model whose parameters the optimizer will update.
    optimizer_name (str): Name of the optimizer (e.g., 'sgd', 'adam', 'rmsprop').
    lr (float): Learning rate.
    momentum (float): Momentum factor (for SGD and RMSprop).
    weight_decay (float): Weight decay (L2 penalty) (for all optimizers).
    beta1 (float): Coefficient used for computing running averages of gradient (for Adam).
    beta2 (float): Coefficient used for computing running averages of gradient's square (for Adam).

    Returns:
    torch.optim.Optimizer: Initialized optimizer.
    """

    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError("Invalid optimizer name")

    return optimizer

