from argparse import ArgumentParser
import torch

def main(args):
    (model_params, first_iter) = torch.load(args.input)
    (active_sh_degree, 
    _xyz, 
    _features_dc, 
    _features_rest,
    _latents,
    color_net_dict,
    _scaling, 
    _rotation, 
    _opacity,
    max_radii2D, 
    xyz_gradient_accum, 
    denom,
    opt_dict, 
    color_net_opt_dict,
    spatial_lr_scale) = model_params

    _features_dc = torch.empty(0)
    _features_rest = torch.empty(0)
    _latents = torch.zeros((_xyz.shape[0], args.latent_size), dtype=torch.float32, device="cuda")

    model_params = (active_sh_degree, 
    _xyz, 
    _features_dc, 
    _features_rest,
    _latents,
    color_net_dict,
    _scaling, 
    _rotation, 
    _opacity,
    max_radii2D, 
    xyz_gradient_accum, 
    denom,
    opt_dict, 
    color_net_opt_dict,
    spatial_lr_scale)

    torch.save((model_params, first_iter), args.output)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--latent_size", type=int, default=48)

    args = parser.parse_args()

    main(args)