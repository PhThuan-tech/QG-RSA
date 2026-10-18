import json
import os
import argparse
import copy
from trainer import RSIAT_train


SUPPORTED_ALGORITHMS = frozenset({"rsiat"})


def validate_algorithm_config(config, require_runtime_assets=True):
    """Validate learner dispatch before any model or dataset is constructed.

    Historical configs did not carry an ``algorithm`` field and continue to
    mean RSIAT.
    """
    validated = copy.deepcopy(config)
    algorithm = str(validated.get("algorithm", "rsiat")).strip().lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            "Unsupported algorithm {!r}; expected one of {}.".format(
                algorithm, sorted(SUPPORTED_ALGORITHMS)
            )
        )

    validated["algorithm"] = algorithm
    return validated

def main():
    args = parse_arguments()
    config = load_json(args.config)
    if args.overlay:
        from utils.qbridge_experiment import merge_dict
        for path in args.overlay:
            config = merge_dict(config, load_json(path))
    merged_config = merge_configs(args, config)
    merged_config = validate_algorithm_config(merged_config)
    RSIAT_train(merged_config)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    """
    Please specify the corresponding JSON file!!!!
    """
    parser.add_argument('--config', type=str, default='./exps/rsiat/imageneta.json',
                        help='Json file of settings.')
    parser.add_argument('--overlay', action='append', default=[], help='Recursively merged JSON override; repeatable')
    return parser.parse_args()

def load_json(settings_path):
    with open(settings_path, encoding="utf-8") as data_file:
        param = json.load(data_file)
    return param

def merge_configs(args, config):
    merged_config = vars(args)  
    merged_config.update(config)  
    return merged_config

if __name__ == '__main__':
    main()
