import sys
sys.path.append('./datasets')
sys.path.append('./datasets/GC')
sys.path.append('./datasets/segment_anything')
sys.path.append('./models')
from train_base import *
from fix_seed import set_seeds

# constants
SYNC = False
GET_MODULE = False
model_gc, model_osn, model_harmonizer, model_enhancer = init_assist_models()

def main():
    args = parse_args()

    # Init dist
    local_rank = 1
    global_rank = 0
    world_size = 1
    if args.fix_seed:
        print('fix seed and use deterministic operations!!')
        # set_seeds(seed=20230506, deterministic=True)
        set_seeds(seed=args.rand_seed, deterministic=True)
    args = init_env(args, local_rank, global_rank)

    model = init_models(args)

    train_sampler, dataloader, train_dataset = init_dataset(args, global_rank, world_size, model_gc, model_osn, model_harmonizer, model_enhancer)
    val_sampler, val_dataloader, val_dataset = init_dataset(args, global_rank, world_size, None, None, None, None, True)

    # model = load_dicts(args, GET_MODULE, model)

    optimizer = init_optims(args, world_size, model)

    lr_scheduler = init_schedulers(args, optimizer, len(dataloader))

    train(args, global_rank, SYNC, GET_MODULE,
          model,
          train_sampler, dataloader, train_dataset, val_sampler, val_dataloader,
          optimizer,
          lr_scheduler
          )


if __name__ == '__main__':
    main()

