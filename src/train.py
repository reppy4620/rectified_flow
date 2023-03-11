import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf

from model import Unet
from sampling import get_sampling_fn
from loss import get_loss_fn
from data import get_data_loader
from utils import seed_everything, Tracker, save_grid
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config_file)

    seed_everything()

    output_dir = Path(f'{cfg.output_dir}')
    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / 'config.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(**cfg.model).to(device)
    optimizer = optim.AdamW(model.parameters(), **cfg.optimizer)

    loss_fn = get_loss_fn(cfg.loss)
    dl = get_data_loader(cfg)
    sampling = get_sampling_fn(cfg.sampling)

    def handle_batch(batch):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(device)

        t = torch.empty(size=(batch_size,), device=device).uniform_(0, 1)
        loss = loss_fn(model, x, t)
        return loss

    noise = torch.randn(25, cfg.model.channels, cfg.image_size, cfg.image_size)
    torch.save(noise, output_dir / 'noise.pth')
    tracker = Tracker(log_file=output_dir / 'loss.csv')
    for epoch in range(1, cfg.n_epoch + 1):
        bar = tqdm(dl, total=len(dl), desc=f'Epoch {epoch}: ')
        model.train()
        for batch in bar:
            optimizer.zero_grad()
            loss = handle_batch(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
            optimizer.step()
            tracker.update(loss=loss.item())
            bar.set_postfix_str(f'Loss: {tracker["loss"].mean():.6f}')
        tracker.write(epoch, clear=True)
        if epoch % cfg.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch:05d}.pth')
        if epoch % cfg.sampling_interval == 0:
            model.eval()
            images = sampling(model, noise)
            save_grid(images, img_dir / f'epoch_{epoch}.png', nrow=5)


if __name__ == '__main__':
    main()
