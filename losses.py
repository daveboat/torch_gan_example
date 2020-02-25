import torch
import torch.nn.functional as F

def discriminator_loss(real_output, fake_output, reduction='mean'):
    """
    The discriminator loss compares the real samples that it classifies against the truth (an array of ones) and the
    fake samples generated by the generator against the truth (an array of zeros).

    The discriminator's loss qualifies how well it can distinguish real samples as real samples, and generated samples
    as generated samples
    """
    real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output), reduction=reduction)
    fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output), reduction=reduction)

    return real_loss + fake_loss

def generator_loss(fake_output, reduction='mean'):
    """
    The generator's loss seeks to make its generated samples be classified as real by the discriminator, so its loss
    compares fake sample classifications against an array of ones

    The generator's loss qualifies how well it can create real-seeming samples
    """
    return F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output), reduction=reduction)