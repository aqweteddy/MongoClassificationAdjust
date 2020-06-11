from torchvision import transforms

from .randaugment import RandAugment


def build_transform(cfg):
    transform = transforms.Compose([
        #     transforms.RandomGrayscale(),
        transforms.Resize(cfg['resize']),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-15, 15)),

        #     transforms.CenterCrop(224),
        #     transforms.RandomResizedCrop(cfg.DATA.RESIZE[1]),
        transforms.ToTensor(),
        transforms.Normalize(cfg['pidxel_mean'],
                             cfg['pixel_std'])])
    return transform


def build_test_transform(cfg):
    # print(cfg)
    transform = transforms.Compose([
        transforms.Resize(cfg['resize']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['pidxel_mean'],
                             cfg['pixel_std'])
    ])
    return transform


def rand_transform(cfg):
    transform = build_transform(cfg)
    # transform.transforms.insert(0, RandAugment(C.get()['randaug']['N'], C.get()['randaug']['M']))
    transform.transforms.insert(0, RandAugment(cfg["rand_n"], cfg["rand_m"]))
    return transform
