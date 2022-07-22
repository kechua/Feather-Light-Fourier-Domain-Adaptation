import torch

from gan.loss.loss_base import Loss


def iteration_loop(batch_x, batch_y, style_enc, gan_model, style_opt, device):
    init_image_x = batch_x['image'].to(device).repeat(1, 3, 1, 1)
    init_image_y = batch_y['image'].to(device).repeat(1, 3, 1, 1)

    style_vec_tuda = style_enc(init_image_x)
    style_vec_list_tuda = [style_vec_tuda[:, k] for k in range(style_vec_tuda.shape[1])]
    image_y_fake, _ = gan_model.generator.forward(
        cond=init_image_x, styles=style_vec_list_tuda
    )

    gan_model.generator_loss([init_image_y], [image_y_fake])\
        .minimize_step(gan_model.optimizer.opt_min, style_opt)

    gan_model.discriminator_train([init_image_y], [image_y_fake.detach()])

    style_opt.zero_grad()
    gan_model.optimizer.opt_min.zero_grad()
    gan_model.optimizer.opt_max.zero_grad()

    del batch_x, batch_y, style_enc, gan_model, style_opt
    torch.cuda.empty_cache()

def tuda_cuda(
        batch_x,
        style_enc_tuda, style_enc_cuda,
        gan_model_tuda, gan_model_cuda,
        style_opt_tuda, style_opt_cuda,
        device, coefs):

    init_image_x = batch_x['image'].to(device).repeat(1, 3, 1, 1)

    style_vec_tuda = style_enc_tuda(init_image_x)
    style_vec_list_tuda = [style_vec_tuda[:, k] for k in range(style_vec_tuda.shape[1])]
    image_y_fake, _ = gan_model_tuda.generator.forward(
        cond=init_image_x, styles=style_vec_list_tuda
    )
   #second iteration
    style_vec_cuda = style_enc_cuda(image_y_fake)
    style_vec_list_cuda = [style_vec_cuda[:, k] for k in range(style_vec_cuda.shape[1])]
    image_x_fake, _ = gan_model_cuda.generator.forward(
        cond=image_y_fake, styles=style_vec_list_cuda
    )

    res: Loss = Loss(
        torch.nn.L1Loss()(image_x_fake, init_image_x) * coefs
    )

    res.minimize_step(
        gan_model_tuda.optimizer.opt_min, gan_model_cuda.optimizer.opt_min,
        style_opt_tuda, style_opt_cuda,
    )


    style_opt_tuda.zero_grad()
    style_opt_cuda.zero_grad()
    gan_model_tuda.optimizer.opt_min.zero_grad()
    gan_model_tuda.optimizer.opt_max.zero_grad()
    gan_model_cuda.optimizer.opt_min.zero_grad()
    gan_model_cuda.optimizer.opt_max.zero_grad()

    del batch_x, style_enc_tuda, style_enc_cuda, gan_model_tuda, gan_model_cuda, style_opt_tuda, style_opt_cuda
    torch.cuda.empty_cache()

    return res.item()