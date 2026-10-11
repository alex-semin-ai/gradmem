"""Idea 3. RMT encoder start memory + GradMem write steps (combined model).

The encoder (a copy of the RMT E1) reads the context in one forward pass and gives h_enc.
GradMem (θ, also a copy of E1) starts its K write steps from M_0 = h_enc, or from
M_0 = W_m h_enc + b with W_m = I, b = 0 at the start (E6p). The read is GradMem's read.
The outer loss trains encoder, θ and the shared token embedding table together, with the
gradient going through the K write steps (grad_mode second, D5).

Plan decisions used here
  D2  two copies of E1 (encoder, GradMem θ), one shared token embedding table
  D3  M_0 = h_enc, optional identity-initialized projection (E6p)
  D6  encoder reconstruction loss off (E6, E7) or on (E6rec, E7rec)
  C1  frozen encoder (no gradient into the encoder)
"""
import torch
from torch import nn

from grad_memgpt import InputPrefixMemoryBackend


class EncoderStartBackend(InputPrefixMemoryBackend):
    """GradMem's prefix memory, but the start memory comes from the encoder when it is set."""

    def init_memory_state(self, batch_size):
        start = getattr(self.owner, '_start_memory', None)
        if start is None:
            return super().init_memory_state(batch_size)
        if start.shape[0] != batch_size:
            raise ValueError(f'start memory batch {start.shape[0]} != batch {batch_size}')
        mem_batch = start
        if not mem_batch.requires_grad:          # frozen encoder: the write steps still need grad w.r.t. memory
            mem_batch = mem_batch.detach().requires_grad_(True)
        return {'mem_batch': mem_batch}, {'mem_batch': mem_batch.detach().clone()}


def encoder_write(encoder, context_input_ids, reconstruction_loss=False):
    """One RMT write pass over [M ; x ; M] (as in rmt.py, K = 1, no control tokens, no projection).

    Returns h_enc (the outputs at the last M positions) and, if asked, RMT's reconstruction loss,
    which needs a second pass that starts from h_enc and predicts the context (rmt.py semantics)."""
    if encoder.n_ctrl_tokens != 0 or encoder.mem_proj_mode != 'none' or encoder.K != 1:
        raise NotImplementedError('encoder_write supports an RMT with K=1, n_ctrl_tokens=0 and mem_proj_mode none')
    B = context_input_ids.size(0)
    M = encoder.n_mem_tokens
    pad_id = encoder.model.config.pad_token_id
    backbone = encoder.model.model
    ctx_emb = encoder.model.get_input_embeddings()(context_input_ids)
    ctx_mask = (context_input_ids != pad_id).to(dtype=torch.long)
    mem_mask = torch.ones(B, M, dtype=torch.long, device=context_input_ids.device)
    attn_mask = torch.cat([mem_mask, ctx_mask, mem_mask], dim=1)
    position_ids = attn_mask.cumsum(-1) - 1

    def one_pass(mem):
        x = torch.cat([mem, ctx_emb, mem], dim=1)
        return backbone(inputs_embeds=x, position_ids=position_ids, attention_mask=attn_mask,
                        output_hidden_states=True, return_dict=True).hidden_states[-1]

    h = one_pass(encoder.mem.unsqueeze(0).expand(B, -1, -1))
    h_enc = h[:, -M:]
    rec_loss = None
    if reconstruction_loss:
        h2 = one_pass(h_enc)
        logits = encoder.model.get_output_embeddings()(h2[:, M - 1:-M])   # last memory slot .. last context token
        labels = context_input_ids.clone()
        labels[labels == pad_id] = -100
        rec_loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                               labels.reshape(-1), ignore_index=-100)
    return h_enc, rec_loss


class EncGradMem(nn.Module):
    """The combined model. Call it like GradMemGPT: forward(input_ids, labels=None, return_mem=False)."""

    def __init__(self, encoder, gradmem, use_projection=False, encoder_reconstruction_loss=False,
                 reconstruction_loss_weight=1.0, freeze_encoder=False):
        super().__init__()
        if gradmem.memory_backend != 'prefix':
            raise ValueError('EncGradMem needs GradMem with memory_backend prefix')
        if encoder.n_mem_tokens != gradmem.n_mem_tokens:
            raise ValueError('encoder and GradMem must have the same number of memory tokens')
        if gradmem.mem_proj_mode != 'none' or gradmem.n_ctrl_tokens != 0:
            raise ValueError('EncGradMem supports GradMem with mem_proj_mode none and n_ctrl_tokens 0')
        if freeze_encoder and encoder_reconstruction_loss:
            raise ValueError('a frozen encoder (C1) cannot use the encoder reconstruction loss, '
                             'its gradient would still reach the shared embedding table')
        self.encoder = encoder
        self.gradmem = gradmem
        # D2: one shared token embedding table; the encoder's output layer stays tied to it
        encoder.model.set_input_embeddings(gradmem.model.get_input_embeddings())
        encoder.model.tie_weights()
        gradmem.memory_backend_impl = EncoderStartBackend(gradmem)
        gradmem._start_memory = None
        d = gradmem.model.get_input_embeddings().weight.shape[1]
        self.proj = None
        if use_projection:                      # D3, E6p: starts as identity, so E6p equals E6 at step 0
            self.proj = nn.Linear(d, d, bias=True)
            with torch.no_grad():
                nn.init.eye_(self.proj.weight)
                self.proj.bias.zero_()
        self.encoder_reconstruction_loss = encoder_reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:                      # C1. The shared embedding table still trains through GradMem
            shared = gradmem.model.get_input_embeddings().weight
            for p in encoder.parameters():
                if p is not shared:
                    p.requires_grad_(False)
        self.main_input_name = 'input_ids'

    def start_memory(self, context_input_ids):
        h_enc, rec_loss = encoder_write(self.encoder, context_input_ids,
                                        reconstruction_loss=self.encoder_reconstruction_loss)
        if self.freeze_encoder:
            h_enc = h_enc.detach()
        start = self.proj(h_enc) if self.proj is not None else h_enc
        return start, rec_loss

    def forward(self, input_ids, labels=None, return_mem=False):
        start, rec_loss = self.start_memory(input_ids['context_input_ids'])
        self.gradmem._start_memory = start
        try:
            out = self.gradmem(input_ids, labels=labels, return_mem=return_mem)
        finally:
            self.gradmem._start_memory = None
        if rec_loss is not None:
            out['inner_loop_stats']['enc_rec_loss'] = rec_loss.detach()
            if labels is not None:
                out['loss'] = out['loss'] + self.reconstruction_loss_weight * rec_loss
        return out


def load_parent(model, state_dict):
    """Load an RMT (E1) state dict, fail on any missing weight except the tied output layer."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith('lm_head.weight')]
    if missing:
        raise RuntimeError(f'{len(missing)} weights not loaded from the parent, e.g. {missing[:5]}')
    return result
