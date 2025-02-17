from neuraldecoding.dataset import Dataset

subject = 'Joker'
date = '2024-08-23'
runs = [1]

dataset = Dataset()

# set the server directory to a custom path - not necessary if the data is in the default path (Z:/ for Windows systems)
# dataset.set_server_directory('F:')

# loading data using the subject-date-runs format
dataset.load_data(subject_name=subject, date=date, runs=runs)

# loading data using the data_path format
# data_path = 'Z:\\Data\\Monkeys\\Joker\\2024-08-23'
# dataset.load_data(data_path=data_path, runs=runs)

# extracting features
features_fields = ['sbp','fingers_kinematics']

features_params = {
    'bin_size': 32,
    'behav_lag': 0,
    'overlap': 0,    
    'remove_first_trial': False,
    'trials_filter': {
        'blank_trial': 0,
        'trial_success': 1,
        'decode_feature': 1
    }
}

features = dataset.extract_features(fields=features_fields, params=features_params)

print('Loading data completed')