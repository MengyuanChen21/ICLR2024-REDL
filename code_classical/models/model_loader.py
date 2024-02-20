import inspect
import torch
from dataset import get_dataset

from models.ModifiedEvidentialN import ModifiedEvidentialNet

create_model = {'menet': ModifiedEvidentialNet}


def load_model(directory_model, name_model, model_type, batch_size_eval=1024):
    model_path = directory_model + name_model
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"

    # Select arguments for model creation
    args = inspect.getfullargspec(create_model[model_type])[0][1:]
    config_dict = torch.load(f'{model_path}', map_location=map_location)['model_config_dict']
    seed = config_dict['seed'] if 'seed' in config_dict.keys() else config_dict['seed_dataset']
    _, _, _, config_dict['N'], _ = get_dataset(config_dict['dataset_name'],
                                            batch_size=config_dict['batch_size'],
                                            split=config_dict['split'],
                                            seed=seed,
                                            test_shuffle_seed=None,
                                            batch_size_eval=batch_size_eval)

    # filtered_config_dict = {arg: config_dict[arg] for arg in args}
    filtered_config_dict = {}
    for arg in args:
        if arg == 'seed' and 'seed' not in config_dict.keys():
            filtered_config_dict['seed'] = config_dict['seed_model']
        else:
            filtered_config_dict[arg] = config_dict[arg]

    # Create model
    model = create_model[model_type](**filtered_config_dict)

    # Load weights
    model.load_state_dict(torch.load(f'{model_path}', map_location=map_location)['model_state_dict'])
    model.eval()

    return model
