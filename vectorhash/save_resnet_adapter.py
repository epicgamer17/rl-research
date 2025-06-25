import torch
import torch.nn as nn
import pretrainedmodels


def main():
    # 1) Choose the Cadene model name
    model_name = 'resnet18'
    # 2) Load Cadene's ResNet-18 with ImageNet weights
    base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    # 3) Extract feature backbone (all layers except the final classifier)
    #    list(base.children()) yields modules in order; drop the last_linear layer
    modules = list(base.children())[:-1]
    feat = nn.Sequential(*modules)

    # 4) Build your projection head (must match PreprocessingCNN latent_dim)
    in_features = base.last_linear.in_features
    proj = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features, 128)
    )

    # 5) Combine into one encoder pipeline
    encoder = nn.Sequential(feat, proj)
    encoder.eval()

    # 6) Save in { 'model_name': ..., 'state_dict': ... } format
    ckpt = {'model_name': model_name, 'state_dict': encoder.state_dict()}
    torch.save(ckpt, 'resnet18_adapter.pth')
    print("Saved ResNet-18 adapter to 'resnet18_adapter.pth'")


if __name__ == '__main__':
    main()
