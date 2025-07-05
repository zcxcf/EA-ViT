import torch
from tqdm import tqdm
import wandb

def eval_stage1(model, valDataLoader, criterion, epoch, optimizer, args, flag, **kwargs):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(**kwargs)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "val/Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "val/Val Acc_" + flag: accuracy})

def eval_stage2(model, valDataLoader, criterion, epoch, optimizer, args, flag, constraint, device):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_constraint_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                t = 1

                model.configure_constraint(constraint=constraint, tau=t)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                attn_mask = torch.mean(attn_mask)
                mlp_mask = torch.mean(mlp_mask)
                embed_mask = torch.mean(embed_mask)
                depth_mlp_mask = torch.mean(depth_mlp_mask)
                depth_attn_mask = torch.mean(depth_attn_mask)

                ce_loss = criterion(preds, label)

                constraint_loss = torch.square(constraint-total_macs)

                loss = ce_loss + constraint_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_constraint_loss += constraint_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_constraint_loss = total_constraint_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            correct = torch.tensor(correct, dtype=torch.float32, device='cuda')
            total = torch.tensor(total, dtype=torch.float32, device='cuda')

            accuracy = correct / total
            print(f"Accuracy: {accuracy.item()}")
            wandb.log({"Epoch": epoch + 1, "val_loss/Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "val_loss/Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "val_loss/Val constraint Loss_" + flag: avg_constraint_loss})

            wandb.log({"Epoch": epoch + 1, "acc/Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "val_attn_mask/Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "val_mlp_mask/Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "val_embed_mask/Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "val_depth_mlp_mask/Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "val_depth_attn_mask/Val depth attn mask_" + flag: val_depth_attn_mask})

            wandb.log({"Epoch": epoch + 1, "val_mac/Val macs_" + flag: val_macs})

            wandb.log({"Epoch": epoch + 1, "val_rate/macs rate_" + flag: accuracy/val_macs})




