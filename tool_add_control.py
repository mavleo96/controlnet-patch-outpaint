import argparse
import os

import torch
from share import *
from cldm.model import create_model

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.input_path), 'Input model does not exist.'
    assert os.path.exists(args.model_path), 'Model path does not exist.'
    assert not os.path.exists(args.output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(args.output_path)), 'Output path is not valid.'

    model = create_model(config_path=args.model_path)

    pretrained_weights = torch.load(args.input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), args.output_path)
    print('Done.')


if __name__ == '__main__':
    main()