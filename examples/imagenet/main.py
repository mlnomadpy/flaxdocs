
import os
import sys
import argparse
import warnings
from typing import Any, List, Optional, Tuple, Callable
from functools import partial

# JAX / Flax / Optax / Sharding
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx
import optax

# Checkpointing & Logging
import orbax.checkpoint as ocp
import wandb

# Data Loading
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from torchvision import transforms
from PIL import Image, PngImagePlugin
from tqdm import tqdm
import numpy as np

# Fix PIL PNG decompression limit
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------------------------
# 1. Neural Network Blocks (NNX)
# -----------------------------------------------------------------------------

class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nnx.Module] = None, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3),
                              strides=stride, padding=1, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3),
                              strides=1, padding=1, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = nnx.relu(out)
        return out


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nnx.Module] = None, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3),
                              strides=stride, padding=1, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv3 = nnx.Conv(out_channels, out_channels * self.expansion,
                              kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_channels * self.expansion, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv3(out)
        out = self.bn3(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = nnx.relu(out)
        return out


class DownsampleBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, rngs):
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                             strides=stride, use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x, training=True):
        x = self.conv(x)
        x = self.bn(x, use_running_average=not training)
        return x


class ResNet(nnx.Module):
    def __init__(self, block_cls, layers: List[int], num_classes: int = 1000,
                 dtype=jnp.float32, rngs: nnx.Rngs = None):
        self.in_channels = 64
        self.dtype = dtype

        self.conv1 = nnx.Conv(3, 64, kernel_size=(7, 7), strides=2, padding=3,
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)

        self.layer1 = self._make_layer(block_cls, 64, layers[0], rngs=rngs)
        self.layer2 = self._make_layer(block_cls, 128, layers[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], stride=2, rngs=rngs)

        self.feature_dim = 512 * block_cls.expansion
        self.head = nnx.Linear(self.feature_dim, num_classes, rngs=rngs)

    def _make_layer(self, block_cls, out_channels, blocks, stride=1, rngs=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block_cls.expansion:
            downsample = DownsampleBlock(self.in_channels, out_channels * block_cls.expansion, stride, rngs)

        layers = []
        layers.append(block_cls(self.in_channels, out_channels, stride, downsample, rngs=rngs))
        self.in_channels = out_channels * block_cls.expansion
        for _ in range(1, blocks):
            layers.append(block_cls(self.in_channels, out_channels, rngs=rngs))
        
        return nnx.List(layers)

    def __call__(self, x, training: bool = True):
        x = x.astype(self.dtype)
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, training=training)

        x = jnp.mean(x, axis=(1, 2))
        return self.head(x)

# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------

class ImageNetStreamDataset(IterableDataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('mlnomad/imagenet-1k-224', split=split, streaming=True)
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            try:
                image = sample['image']
                label = sample['label']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                yield image, label
            except Exception:
                continue

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_numpy_batch(batch):
    images, labels = batch
    images_np = images.numpy().transpose(0, 2, 3, 1)
    labels_np = labels.numpy()
    return images_np, labels_np

# -----------------------------------------------------------------------------
# 3. Training Step (NNX + Mesh)
# -----------------------------------------------------------------------------

@nnx.jit
def train_step(model, optimizer, batch_images, batch_labels):
    
    def loss_fn(model):
        outputs = model(batch_images, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
        
        acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
        return loss, acc

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, acc

@nnx.jit
def val_step(model, batch_images, batch_labels):
    outputs = model(batch_images, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
        
    acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
    return loss, acc

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='./checkpoints_flax')
    parser.add_argument('--checkpoint-keep', type=int, default=3)
    parser.add_argument('--wandb-project', type=str, default="imagenet-flax")
    parser.add_argument('--wandb-entity', type=str, default="irf-sic")
    parser.add_argument('--wandb-name', type=str, default=None)
    
    args, unknown = parser.parse_known_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))

    ckpt_dir = os.path.abspath(args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=args.checkpoint_keep, create=True)
    mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options)

    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), ('data',))
    data_sharding = NamedSharding(mesh, P('data', None, None, None)) 
    label_sharding = NamedSharding(mesh, P('data'))
    replicated_sharding = NamedSharding(mesh, P())

    rngs = nnx.Rngs(0)
    block_map = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
    }
    block_cls, layers = block_map[args.model]
    dtype = jnp.bfloat16 if args.mixed_precision else jnp.float32

    with mesh:
        model = ResNet(block_cls=block_cls, layers=layers, num_classes=1000, dtype=dtype, rngs=rngs)
        schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.epochs * 1000)
        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=schedule, weight_decay=0.01), wrt=nnx.Param)

        state = nnx.state((model, optimizer))
        sharded_state = jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf, replicated_sharding), state)
        nnx.update((model, optimizer), sharded_state)

    train_dataset = ImageNetStreamDataset(split='train', transform=get_transforms(True))
    val_dataset = ImageNetStreamDataset(split='validation', transform=get_transforms(False))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    best_acc = 0.0
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_accs = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            global_step += 1
            imgs_np, lbls_np = get_numpy_batch(batch)
            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)
            
            loss, acc = train_step(model, optimizer, imgs_sharded, lbls_sharded)
            
            loss_val, acc_val = float(loss), float(acc)
            epoch_losses.append(loss_val)
            epoch_accs.append(acc_val)
            
            if args.wandb_project:
                wandb.log({'train/iter_loss': loss_val, 'train/iter_acc': acc_val, 'trainer/global_step': global_step, 'epoch': epoch})

            pbar.set_postfix({'acc': f"{np.mean(epoch_accs[-10:]):.4f}"})

        # Validation
        model.eval()
        total_acc, total_loss = [], []
        for batch in tqdm(val_loader, desc='Val'):
            imgs_np, lbls_np = get_numpy_batch(batch)
            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)
            loss, acc = val_step(model, imgs_sharded, lbls_sharded)
            total_acc.append(acc)
            total_loss.append(loss)
        
        val_acc = np.mean(total_acc) * 100
        val_loss = np.mean(total_loss)
        print(f"Epoch {epoch} Val Acc: {val_acc:.2f}%")
        
        if args.wandb_project:
            wandb.log({'epoch': epoch, 'train/loss': np.mean(epoch_losses), 'val/loss': val_loss, 'val/acc': val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            raw_state = nnx.state((model, optimizer))
            mngr.save(step=epoch, args=ocp.args.StandardSave(raw_state))
            print(f"Saved new best model: {best_acc:.2f}%")

    if args.wandb_project:
        wandb.finish()

if __name__ == '__main__':
    main()
