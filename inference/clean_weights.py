import argparse
import typing
import torch

if __name__ == '__main__':
     # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        '-w',
        '--weights_path',
        type=str,
        help='The weights path to clean')
    arg_parser.add_argument(
        '-o',
        '--output_weights_path',
        type=str,
        help='The weights path cleaned')
    args = arg_parser.parse_args()

    new_weights = {
        'model_state_dict': torch.load(args.weights_path)['model_state_dict']
    }
    torch.save(new_weights, args.output_weights_path)