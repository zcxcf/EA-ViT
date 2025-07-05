import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import wandb

from config import get_args_parser
from dataloader.image_datasets import build_image_dataset
from models.model_stage1 import EAViTStage1
from utils.eval_flag import eval_stage1
from utils.lr_sched import adjust_learning_rate
from utils.set_wandb import set_wandb

flags_list = ['l', 'm', 's', 'ss', 'sss']

mlp_ratio_list = [4, 4, 3, 3, 2, 1, 0.5]

mha_head_list = [12, 11, 10, 9, 8, 7, 6]

eval_mlp_ratio_list = [4, 3, 2, 1, 0.5]

eval_mha_head_list = [12, 11, 10, 8, 6]


def train(args):
    torch.cuda.set_device(args.device)
    dataset_train, dataset_val, nb_classes = build_image_dataset(args)

    trainDataLoader = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valDataLoader = DataLoader(dataset_val, args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = EAViTStage1(embed_dim=768, depth=12, mlp_ratio=4, num_heads=12, num_classes=nb_classes,
                               drop_path_rate=args.drop_path, qkv_bias=True)
    model.to(args.device)

    if args.pretrained:
        checkpoint_path = args.rearranged_checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    folder_path = 'logs_weight/stage_1/'+args.dataset

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args, name='EA-ViT_stage1')

    current_stage = 0

    for epoch in range(args.epochs):

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            adjust_learning_rate(optimizer, epoch+1, args)

            model.train()
            total_loss = 0

            if epoch in args.curriculum_epochs:
                stage_index = args.stage_epochs.index(epoch)
                wandb.log({"Epoch": epoch + 1, "stage": stage_index})
                current_stage += 1

            wandb.log({"Epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]['lr']})

            for batch_idx, (img, label) in enumerate(trainDataLoader):

                img = img.to(args.device)
                label = label.to(args.device)
                optimizer.zero_grad()

                loss = 0

                r = random.randint(0, current_stage)
                mlp_ratio = mlp_ratio_list[r]

                r = random.randint(0, current_stage)
                mha_head = mha_head_list[r]

                r = random.randint(0, current_stage)
                sub_dim = 64 * mha_head_list[r]

                depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                r = random.randint(0, 5)
                if r > 2:
                    r = 0
                if r > 0:
                    num_to_remove = random.choice(list(range(r)))
                    indices_to_remove = random.sample(range(len(depth_list)), num_to_remove)
                    depth_list = [depth_list[i] for i in range(len(depth_list)) if i not in indices_to_remove]

                model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head)

                preds = model(img)
                loss += criterion(preds, label)

                if batch_idx % 10 == 0:
                    wandb.log({"train/train Batch Loss": loss.item()})
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)
                break


            epoch_loss = total_loss / len(trainDataLoader)
            print("train loss", epoch_loss)
            wandb.log({"Epoch": epoch + 1, "train/Train epoch Loss": epoch_loss})

            pbar.close()

        if epoch % 2 == 0:
            for index, f in enumerate(flags_list):
                mlp_ratio = eval_mlp_ratio_list[index]
                mha_head = eval_mha_head_list[index]
                sub_dim = 64 * eval_mha_head_list[index]
                depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

                eval_stage1(model, valDataLoader, criterion, epoch, optimizer, args, flag=f, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head)

        torch.save(model.state_dict(), weight_path+'/stage1.pth')
        print('save')

if __name__ == '__main__':
    import os
    os.environ["WANDB_MODE"] = "offline"

    args = get_args_parser()
    train(args)




