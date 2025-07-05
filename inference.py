import re

from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile
from timm.loss import LabelSmoothingCrossEntropy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_args_parser
from dataloader.image_datasets import build_image_dataset
from models.model_stage2 import EAViTStage2, ModifiedBlock

def search_num(strings):
    match = re.search(r"[-+]?\d*\.\d+|\d+", strings)
    if match:
        number = float(match.group())
        if 'MMACs' in strings:
            number = number/1000
        # print(number)
    else:
        number = 0
        print("no number")
    return number


def eval_dynamic(model, valDataLoader, device):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description('eval')

        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.update(1)

            accuracy = 100.0 * correct / total
            print()
            print("val acc", accuracy)

            pbar.close()
    return accuracy

def caculate_macs(model, device):
    with get_accelerator().device(device):
        flops, macs, params = get_model_profile(model=model, # model
                                        input_shape=(1, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=False, # prints the model graph with the measured profile attached to each module
                                        detailed=False, # print the detailed profile
                                        module_depth=0, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=10, # the number of warm-ups before measuring the time of each module
                                        as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling

        # print("flops", flops)
        print("macs", macs)
        # print("params", params)
    macs = search_num(macs)
    return macs


if __name__ == '__main__':
    args = get_args_parser()
    dataset_train, dataset_val, nb_classes = build_image_dataset(args)

    valDataLoader = DataLoader(dataset_val, args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    device = args.device

    model = EAViTStage2(embed_dim=768, depth=12, mlp_ratio=4, num_heads=12, num_classes=nb_classes,
                                 drop_path_rate=args.drop_path, qkv_bias=True, block=ModifiedBlock)

    model_path = args.stage2_checkpoint_path
    para = torch.load(model_path, map_location=device)
    model.load_state_dict(para, strict=False)
    model = model.to(device)
    model.eval()


    input_tensor = torch.randn(1, 3, 224, 224).to(device=device)

    flag = None

    constraint = torch.tensor(args.constraint).to(device).unsqueeze(0)

    model.configure_constraint(constraint=constraint, tau=1)

    #
    # x, mean_attn_mask_value, mean_mlp_mask_value, mean_embed_dim_mask, mean_depth_attn_mask, mean_depth_mlp_mask, total_macs = model(input_tensor)
    #
    # print("macs", total_macs)

    # mask_attn = []
    # mask_mlp = []
    # #
    # embed_mask = torch.tensor([1.0]*12+[0.0]*0).to(device)
    #
    # for i in range(12):
    #     mask_attn.append(torch.tensor([1.0]*12+[0.0]*0).to(device))
    #
    # for i in range(12):
    #     mask_mlp.append(torch.tensor([1.0]*8+[0.0]*0).to(device))
    #
    # depth_attn_mask = torch.tensor([1.0]*12+[0.0]*0).to(device)
    # depth_mlp_mask = torch.tensor([1.0]*12+[0.0]*0).to(device)
    #
    # model.set_mask(embed_mask, mask_attn, mask_mlp, depth_attn_mask, depth_mlp_mask)

    caculate_macs(model, device=device)
    eval_dynamic(model, valDataLoader, device=device)


