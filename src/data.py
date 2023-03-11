from datasets import load_dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader


def basic(cfg):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])
    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert(cfg.image_convert)) for image in examples[cfg.image_key]]
        del examples[cfg.image_key]
        return examples

    ds = load_dataset(cfg.ds)
    transformed_ds = ds.with_transform(transforms).remove_columns("label")

    dl = DataLoader(transformed_ds["train"], batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    return dl


def afhq_cat(cfg):
    transform = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])
    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert(cfg.image_convert)) for image in examples[cfg.image_key]]
        del examples[cfg.image_key]
        return examples

    ds = load_dataset(cfg.ds)
    transformed_ds = ds.filter(lambda x: x["label"] == 0).with_transform(transforms).remove_columns("label")

    dl = DataLoader(transformed_ds["train"], batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    return dl


_dl_dict = {
    'basic': basic,
    'afhq_cat': afhq_cat
}

def get_data_loader(cfg):
    return _dl_dict[cfg.data_loader](cfg)
