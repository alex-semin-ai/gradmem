"""Idea 3 on MQAR. RMT encoder start memory + GradMem write steps, trained jointly (E6, E7, E6rec, E7rec, E6p, C1).

Both parts start from the same RMT checkpoint (E1, --init_checkpoint). The encoder reads the context in one
forward pass and gives h_enc, GradMem starts its K write steps from M_0 = h_enc (or from an identity-initialized
projection of it), and the read and the outer loss are GradMem's. See enc_gradmem.py and the plan, D2 to D6.

Data, metrics, the stop rule and the Trainer setup are those of run_gradmemgpt_on_mqar.py, so the numbers are
directly comparable with E2r, E3r and E8r. Differences
  - --init_checkpoint is required and must be an RMT checkpoint (it fills the encoder and GradMem's theta).
  - Checkpoints are saved as pytorch_model.bin, because the shared embedding table appears under several
    names and safetensors refuses shared tensors (plan, Q8).
  - The encoder's reconstruction loss, when on, is logged as enc_rec_loss (training and eval).
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import accelerate
import torch
import transformers
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback, HfArgumentParser, TrainingArguments

import run_gradmemgpt_on_mqar as G
from enc_gradmem import EncGradMem, load_parent
from grad_memgpt import GradMemGPTConfig
from llama_config import llama_3_2_1b_config
from rmt import RMT2SegmConfig
from run_rmt_on_mqar import MQARRMT
from zoology_mqar_data import ZOOLOGY_MQAR_SOURCE, build_mqar_datasets

# the encoder's reconstruction loss is logged with the other loss parts (CustomTrainer and compute_metrics_fn
# read this list at run time)
if 'enc_rec_loss' not in G.TRAIN_COMPONENT_KEYS:
    G.TRAIN_COMPONENT_KEYS.append('enc_rec_loss')


@dataclass
class CombinedArgs(G.ExperimentArgs):
    use_projection: Optional[bool] = field(
        default=False, metadata={'help': 'E6p. M_0 = W h_enc + b with W = I and b = 0 at the start (D3)'})
    encoder_reconstruction_loss: Optional[bool] = field(
        default=False, metadata={'help': "E6rec, E7rec. Add the encoder's RMT reconstruction loss (D6)"})
    encoder_reconstruction_loss_weight: Optional[float] = field(default=1.0)
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={'help': 'C1. No gradient into the encoder (the shared embedding table still trains)'})


def base_llama_config(args):
    config = llama_3_2_1b_config()
    config.num_hidden_layers = args.n_layer
    config.num_attention_heads = args.n_head
    config.num_key_value_heads = args.n_head
    config.hidden_size = args.n_embd
    config.head_dim = config.hidden_size // config.num_attention_heads
    config.intermediate_size = config.hidden_size * 4
    config.rope_scaling = None
    config.rope_theta = 10000.0
    config.max_position_embeddings = args.max_position_embeddings
    config.torch_dtype = 'float32'
    config.vocab_size = args.vocab_size
    config.pad_token_id = 0
    config.bos_token_id = None
    config.eos_token_id = None
    config.use_cache = False
    return config


def check_args(args):
    problems = []
    if args.init_checkpoint is None:
        problems.append('--init_checkpoint (the RMT parent, E1) is required')
    if args.pretrained_model is not None or args.init_base_checkpoint is not None:
        problems.append('--pretrained_model and --init_base_checkpoint are not used, the parent is --init_checkpoint')
    if args.base_model != 'llama':
        problems.append('only --base_model llama is supported')
    if args.memory_backend != 'prefix':
        problems.append('only --memory_backend prefix is supported')
    if args.mem_proj_mode != 'none' or args.use_mem_proj or args.n_ctrl_tokens != 0:
        problems.append('GradMem must use mem_proj_mode none, use_mem_proj false and n_ctrl_tokens 0')
    if args.freeze_encoder and args.encoder_reconstruction_loss:
        problems.append('a frozen encoder (C1) cannot use the encoder reconstruction loss')
    if problems:
        raise ValueError('Invalid arguments. ' + '; '.join(problems))


def gradmem_config_from_args(args, config):
    """GradMem's config exactly as run_gradmemgpt_on_mqar.py builds it (the read loss at the query positions)."""
    return GradMemGPTConfig(
        pretrained_model=None, base_config=copy.deepcopy(config), memory_backend=args.memory_backend,
        n_mem_tokens=args.n_mem_tokens, K=args.K, last_K_second_order=args.last_K_second_order,
        lr=args.inner_lr, use_adam=args.use_adam, grad_mode=args.grad_mode, n_ctrl_tokens=args.n_ctrl_tokens,
        inner_clip_value=args.inner_clip_value, inner_clip_norm=args.inner_clip_norm,
        use_mem_proj=args.use_mem_proj, mem_proj_mode=args.mem_proj_mode,
        use_write_head=args.use_write_head, use_write_lora=args.use_write_lora,
        write_lora_r=args.write_lora_r, write_lora_alpha=args.write_lora_alpha,
        write_lora_dropout=args.write_lora_dropout, write_lora_target_modules=args.write_lora_target_modules,
        lora_mem_placement=args.lora_mem_placement, lora_mem_r=args.lora_mem_r, lora_mem_alpha=args.lora_mem_alpha,
        lora_mem_dropout=args.lora_mem_dropout, lora_mem_layers=args.lora_mem_layers,
        lora_mem_target_modules=args.lora_mem_target_modules, kv_mem_layers=args.kv_mem_layers,
        freeze_backbone=args.freeze_backbone, use_gradient_checkpointing=args.use_gradient_checkpointing,
        attn_implementation=args.attn_implementation, write_objective=args.write_objective,
        energy_head_hidden_dim=args.energy_head_hidden_dim,
        write_reconstruction_weight=args.write_reconstruction_weight, write_energy_weight=args.write_energy_weight,
        energy_rank_weight=args.energy_rank_weight, energy_traj_weight=args.energy_traj_weight,
        energy_margin=args.energy_margin, energy_traj_margin=args.energy_traj_margin,
        energy_rank_temperature=args.energy_rank_temperature, energy_mix_alpha=args.energy_mix_alpha,
        energy_anchor_weight=args.energy_anchor_weight, add_inner_loss_to_outer=args.add_inner_loss_to_outer,
        inner_loss_weight=args.inner_loss_weight, read_loss_alignment='query_position',
        mqar_vocab_size=args.vocab_size, mqar_dense_queries=args.dense_queries)


def encoder_config_from_args(args, config):
    """The encoder is a plain RMT x1 like E1 (no control tokens, no projection, no write head)."""
    return RMT2SegmConfig(
        pretrained_model=None, base_config=copy.deepcopy(config), n_mem_tokens=args.n_mem_tokens, K=1,
        n_ctrl_tokens=0, use_mem_proj=False, mem_proj_mode='none', use_reconstruction_loss=True,
        reconstruction_loss_weight=1.0, attn_implementation=args.attn_implementation,
        read_loss_alignment='query_position', mqar_vocab_size=args.vocab_size, mqar_dense_queries=args.dense_queries)


def build_model(args, load_parent_weights=True):
    """The combined model. Both parts are loaded from the RMT parent unless load_parent_weights is False
    (eval_runs.py loads a trained checkpoint of the combined model instead)."""
    config = base_llama_config(args)
    encoder = MQARRMT(encoder_config_from_args(args, config))
    gradmem = G.MQARGradMemGPT(gradmem_config_from_args(args, config))
    if load_parent_weights:
        state_dict = load_file(args.init_checkpoint)
        if state_dict['mem'].shape != gradmem.mem.shape:
            raise ValueError(f'The parent has memory {tuple(state_dict["mem"].shape)}, '
                             f'the model expects {tuple(gradmem.mem.shape)}.')
        load_parent(encoder, state_dict)   # raises on any missing weight except the tied output layer
        load_parent(gradmem, state_dict)
    return EncGradMem(encoder, gradmem, use_projection=args.use_projection,
                      encoder_reconstruction_loss=args.encoder_reconstruction_loss,
                      reconstruction_loss_weight=args.encoder_reconstruction_loss_weight,
                      freeze_encoder=args.freeze_encoder)


if __name__ == '__main__':
    parser = HfArgumentParser(CombinedArgs)
    args = parser.parse_args_into_dataclasses()[0]
    check_args(args)

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    transformers.utils.logging.set_verbosity(G.log_lvl)
    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')

    source_train, source_valid, train_data_seed, valid_data_seed = build_mqar_datasets(
        vocab_size=args.vocab_size, input_seq_len=args.input_seq_len, num_kv_pairs=args.num_kv_pairs,
        train_num_examples=args.train_num_examples, valid_num_examples=args.valid_num_examples,
        power_a=args.power_a, random_non_queries=args.random_non_queries, data_seed=args.data_seed,
        dense_queries=args.dense_queries, query_sampling=args.query_sampling)
    context_size = args.num_kv_pairs * 2
    train_dataset = G.GradMemMQARDataset(source_train, context_size=context_size)
    valid_dataset = G.GradMemMQARDataset(source_valid, context_size=context_size)
    output_dir = Path(args.exp_path)

    if accel.is_main_process:
        run_config = {
            'cli_args': dict(vars(args)),
            'task_source': ZOOLOGY_MQAR_SOURCE,
            'task': 'Zoology MQAR, RMT encoder start memory + GradMem write steps (Idea 3)',
            'context_size': context_size,
            'query_size': args.input_seq_len - context_size,
            'train_data_seed': train_data_seed,
            'valid_data_seed': valid_data_seed,
            'init_rmt_checkpoint': args.init_checkpoint,
            'start_memory': 'W h_enc + b, W = I and b = 0 at the start' if args.use_projection else 'h_enc',
            'encoder_reconstruction_loss': args.encoder_reconstruction_loss,
            'freeze_encoder': args.freeze_encoder,
        }
        logger.info(f'saving experiment configuration to {args.exp_path}')
        output_dir.mkdir(parents=True)
        json.dump(run_config, open(output_dir / 'config.json', 'w'), indent=4)

    model = build_model(args)
    if accel.mixed_precision == 'bf16':
        model.to(torch.bfloat16)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    logger.info(f'model: {model}')
    logger.info(f'parameters {n_all}, trainable {n_train}')
    logger.info(f'train examples: {len(train_dataset)}; valid examples: {len(valid_dataset)}')

    data_collator = G.collate_fn if torch.cuda.is_available() else G.collate_fn_numpy
    args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    if args.total_batch_size is None:
        args.total_batch_size = args_total_bs
    assert args.total_batch_size == args_total_bs

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.use_gradient_checkpointing,
        eval_strategy='steps',
        save_strategy='steps',
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,
        include_for_metrics=['inputs'],
        save_total_limit=1,
        save_safetensors=False,   # the shared embedding table appears under several names (plan, Q8)
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = G.CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=G.compute_metrics_fn,
        preprocess_logits_for_metrics=G.preprocess_logits_for_metrics,
        callbacks=[G.ConsoleEvalMetricsCallback(),
                   EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   G.StopOnMetricValue(metric_name='all_queries_exact_match', value=args.stop_on_metric_value,
                                       higher_is_better=True, consecutive=args.stop_on_metric_consecutive)],
    )
    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(valid_dataset)
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')
