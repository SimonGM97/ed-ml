from config.params import Params
import argparse
import inspect
import string


ARGS = {
    'data_processing.py': [
        'raw_df', 
        'save'
    ],
    'model_tuning.py': [
        'algorithms',
        'eval_metric',
        'val_splits',
        'train_test_ratio',
        'n_candidates',
        'local_registry',
        'max_evals',
        'timeout_mins',
        'loss_threshold',
        'min_performance'
    ],
    'inference.py': [
        'course_name',
        'user_uuids',
        'course_uuids',
        'particion',
        'pick_random'
    ]
}


def parse_type(value: str):
    # None
    if value.lower() == 'none':
        return None
    
    # True
    if value.lower() == 'true':
        return True
    
    # False
    if value.lower() == 'false':
        return False

    # float
    if '.' in value:
        return float(value)
    
    # int
    if value[0] in '1234567890':
        return int(value)
    
    # list
    if ', ' in value:
        return value.split(', ')
    
    return value

def add_parser_arguments(parser: argparse.ArgumentParser):
    # Find file name of the file that triggered the function
    frame = inspect.currentframe()
    file_name = frame.f_back.f_code.co_filename.split('/')[-1]

    for arg_name in ARGS[file_name]:
        parser.add_argument(
            f'--{arg_name}',
            type=parse_type,
            default=getattr(Params, arg_name)
        )
    
    return parser
