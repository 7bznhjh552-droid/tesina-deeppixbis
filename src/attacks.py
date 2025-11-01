import torch

def fgsm_attack(model, loss_fn, images, labels, epsilon=0.01):
    """
    Aplica FGSM generando ejemplos adversarios.
    Asegura compatibilidad con modelos que devuelven mapas (ej. DeepPixBiS).
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)

    # Si la salida tiene forma (B,1,H,W) o (B,H,W), reducir a un escalar por imagen
    if outputs.ndim == 4:
        outputs = outputs.squeeze(1)
    if outputs.ndim == 3:
        outputs = outputs.mean(dim=[1, 2])

    loss = loss_fn(outputs, labels.float())
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    adv_images = images + epsilon * grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()
