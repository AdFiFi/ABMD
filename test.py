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
os.environ['WANDB_MODE'] = "offline"

task_id = 4


def cross_subject(args):
    model = f"{args.framework if args.framework else 'SM'}_{args.modality}" \
            f"{f'_{args.brain_encoder}_{args.brain_fusion}' if args.brain_encoder else ''}" \
            f"{f'-{args.text_encoder}_{args.text_fusion}' if args.text_encoder else ''}" \
            f"{f'-{args.vision_encoder}_{args.vision_fusion}' if args.vision_encoder else ''}" \
            f"{f'-{args.audio_encoder}_{args.audio_fusion}' if args.audio_encoder else ''}"
    if args.do_train:
        results = Recorder()
        group_name = f"{args.dataset}_{model}_{args.train_percent}_{args.batch_size}" \
                     f"{f'-{args.extra_info}' if args.extra_info else ''}"

        run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                         group=f"{group_name}",
                         tags=[args.dataset, args.framework if args.framework else "SM", args.modality])

        trainer = eval(f"{args.framework}Trainer")(args, task_id=task_id)
        init_logger(f'{args.log_dir}/train_{model}_{args.dataset}.log')
        logger.info(f"{'#'*10} Repeat:{task_id} {'#'*10}")
        if args.two_step:
            trainer.two_step_train()
        else:
            trainer.train()
        results.add_record(trainer.get_best())
        run.finish()
        results.save(os.path.join(args.model_dir, args.modality, model, 'results.json'))
    if args.do_evaluate:
        group_name = f"{args.dataset}_{model}_{args.train_percent}_{args.batch_size}" \
                     f"{f'-{args.extra_info}' if args.extra_info else ''}"

        run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                         group=f"{group_name}",
                         tags=[args.dataset, args.framework if args.framework else "SM", args.modality])

        trainer = eval(f"{args.framework}Trainer")(args, task_id=task_id)
        trainer.load_model()
        # result = trainer.multiple_evaluate(True)
        result = trainer.multiple_evaluate()
        trainer.save_hidden_state()
        run.finish()


if __name__ == '__main__':
    Args = init_config()
    cross_subject(Args)
