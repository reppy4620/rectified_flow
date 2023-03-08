import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf

from model import Unet
from sampling import sample_ode
from utils import seed_everything, save_grid
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_generate', type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gen_dir = output_dir / 'gen'
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = OmegaConf.load(output_dir / 'config.yaml')

    seed_everything()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(**cfg.model).to(device)
    ckpt_file = list(sorted((output_dir / 'ckpt').glob('*.pth')))[-1]
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model'])
    model.eval()
    epoch = ckpt['epoch']

    for n in tqdm(range(args.num_generate)):
        noise = torch.randn(25, cfg.model.channels, cfg.image_size, cfg.image_size)
        images = sample_ode(model, noise)
        save_grid(images, gen_dir / f'gen-{n+1}_epoch_{epoch}.png', nrow=5)


if __name__ == '__main__':
    main()
