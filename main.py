import torch
import os
from models.efficienet import efficientnet, EffNetTrainer
from datasets.dataloader import make_test_loader
# from config import hparams
from tqdm import tqdm
import json
import logging
from argparse import  ArgumentParser


GPU_ID = '2'
logger = logging.getLogger('Inferer')
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def load_config(config):
    with open(config) as f:
        return json.load(f)

def load_models_loaders(folder):
    models_loaders = [] # [(model, loader), ]

    for ckpt in os.listdir(folder):
        ckpt = os.path.join(folder, ckpt)
        if '.pth' in ckpt[-4:]: # normal
            logging.info(f'load model: {ckpt}')
            model = efficientnet('b7').to('cuda').eval()
            model.load_state_dict(torch.load(ckpt, map_location='cuda:0'))
            loader = make_test_loader(load_config(ckpt.replace('.pth', '.json')))
        elif '.ckpt' in ckpt[-5:]: # light
            logging.info(f'load model: {ckpt}')
            model = EffNetTrainer.load_from_checkpoint(ckpt).to('cuda').eval()
            loader = make_test_loader(load_config(ckpt.replace('.ckpt', '.json')))
        else:
            continue
        yield model, loader


def predict(model, loader):
    result = []
    targets = []
    for img, target in tqdm(loader):
        img = img.to('cuda')
        logits = model(img)[:,:3]
        logits = torch.softmax(logits, dim=1).detach().cpu().tolist()
        result.extend(logits)
        targets.append(target)
    
    return torch.tensor(result, requires_grad=False), targets

def get_weighted_result(results, weights):
    """
    results[torch.tensor]: [num_models, num_samples, num_classes]
    weights[list]: weight for each model [num_models]
    """
    assert results.shape[0] == len(weights)
    weights = torch.tensor(weights)
    weighted_sum = (weights[:, None, None] * results).sum(0)
    # print(weighted_sum.shape)
    return weighted_sum

if __name__ == '__main__':
    # test_loader = make_test_loader(hparams)
    parser = ArgumentParser(prog='Inferer')
    parser.add_argument('--gpu_id', type=int, nargs='?', default=1,
                        help='set GPU id (default: 1)')
    parser.add_argument('--weights',  type=float, nargs='+')

    args = parser.parse_args()
    GPU_ID = args.gpu_id
    weights = args.weights

    results = None # dim: num_samples, num_classes
    models_loaders = load_models_loaders('weights')
    for idx, (model, loader) in enumerate(models_loaders):
        logger.info(f'No. {idx + 1}')
        with torch.no_grad():
            result, _ = predict(model, loader)
        
        if results is not None:
            results = torch.stack([results, result], axis=0)
        else:
            results = result
        del model
        del loader
        torch.cuda.empty_cache()

    
    results = get_weighted_result(results, weights)
    value, idx = results.topk(1, dim=-1)
    idx = idx.squeeze(-1).tolist()

    loader = make_test_loader(load_config('/home/dl406410029/mango_emsam/weights/efficient_b7_83.json'))
    ans = []
    for inp, target in loader:
        ans.extend(target.tolist())
    
    correct = [1 for a, b in zip(idx, ans) if a == b]
    print(f'{correct.count(1) / len(ans) * 100:.5f}')
    print(f'{correct.count(1)} / {len(correct)}')

    # results = torch.tensor(results, requires_grad=False)
    