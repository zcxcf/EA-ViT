import wandb

def set_wandb(args, name):
    wandb.init(
        project=name,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        name=args.stage+args.dataset+'_lr'+str(args.lr)+str(args.min_lr)+'_epoch'+str(args.epochs)
    )

