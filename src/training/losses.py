import torch.nn.functional as functional


def masked_cross_entropy_loss(logits, tag_ids, mask):
    tag_count = logits.size(-1)
    flat_logits = logits.reshape(-1, tag_count)
    flat_tag_ids = tag_ids.reshape(-1)
    flat_mask = mask.reshape(-1)

    active_logits = flat_logits[flat_mask]
    active_tag_ids = flat_tag_ids[flat_mask]
    return functional.cross_entropy(active_logits, active_tag_ids)


def masked_language_model_loss(logits, target_ids, mask):
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target_ids = target_ids.reshape(-1)
    flat_mask = mask.reshape(-1)

    active_logits = flat_logits[flat_mask]
    active_target_ids = flat_target_ids[flat_mask]
    return functional.cross_entropy(active_logits, active_target_ids)
