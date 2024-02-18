import os

import wandb

from config import init_config
from trainers import *
from utils import *
from torch import distributed


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"
logger = logging.getLogger(__name__)
os.environ['WANDB_API_KEY'] = "local-1c1ed7bea12024a133bfee72d76ba71b8ca17e4f"
# os.environ['WANDB_MODE'] = "offline"

task_id = 2
subject_id = 1


def within_subject(args):
    model = f"{args.framework if args.framework else 'SM'}_{args.modality}" \
            f"{f'_{args.brain_encoder}_{args.brain_fusion}' if args.brain_encoder else ''}" \
            f"{f'-{args.text_encoder}_{args.text_fusion}' if args.text_encoder else ''}" \
            f"{f'-{args.vision_encoder}_{args.vision_fusion}' if args.vision_encoder else ''}" \
            f"{f'-{args.audio_encoder}_{args.audio_fusion}' if args.audio_encoder else ''}"
    if args.do_train:
        local_rank = 0
        group_name = ''
        if args.do_parallel:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            distributed.init_process_group('nccl', world_size=world_size, rank=rank)
            # distributed.init_process_group('gloo', world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(local_rank)
        best_results = Recorder()
        group_name = f"{args.dataset}-{model}-{args.train_percent}-{args.batch_size}" \
                     f"{args.extra_info}"

        run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                         group=f"{group_name}",
                         tags=[args.dataset, args.framework if args.framework else "SM",
                               args.modality, f'id_{subject_id}'])

        trainer = eval(f"{args.framework}Trainer")(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        init_logger(f'{args.log_dir}/train_{args.framework}-{model}_{args.dataset}.log')
        logger.info(f"{'#'*10} Subject:{task_id} {'#'*10}")
        if args.two_step:
            trainer.two_step_train()
        else:
            trainer.train()
        best_results.add_record(trainer.get_best())
        run.finish()
        best_results.save(os.path.join(args.model_dir, args.modality, model, 'best_results.json'))
        run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                         group=f"{group_name}-results", tags=[args.dataset])
        wandb.log({f"best {k}": v for k, v in best_results.get_avg().items()})
        run.finish()


if __name__ == '__main__':
    Args = init_config()
    within_subject(Args)
