import torch


def token_accuracy(logits, tag_ids, mask):
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        active_predictions = predictions[mask]
        active_tags = tag_ids[mask]

        if active_tags.numel() == 0:
            return 0.0

        correct_count = (active_predictions == active_tags).sum().item()
        return correct_count / active_tags.numel()
