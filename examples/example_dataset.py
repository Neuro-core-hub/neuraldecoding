import matplotlib.pyplot as plt
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
features_fields = ['sbp','mav','fingers_kinematics']  # sbp and mav are the same thing, just different names for the same feature when used for brain or EMG

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

# plot for debugging feature extraction

plt.close('all')
fig, axs = plt.subplots(4,1, figsize=(15, 10))

# Plot EMG channel 1
axs[0].plot(features['mav'][:1000, 32])
axs[0].set_title('MAV-EMG Channel 1')

# Plot EMG channel 7
axs[1].plot(features['mav'][:1000, 44])
axs[1].set_title('MAV-EMG Channel 6')

# Plot kinematics position (index 1 and 3)
axs[2].plot(features['fingers_kinematics'][:1000, 1])
axs[2].plot(features['fingers_kinematics'][:1000, 3])
axs[2].set_title('Kinematics Position')

# Plot kinematics velocity (index 6 and 9)
axs[3].plot(features['fingers_kinematics'][:1000, 6])
axs[3].plot(features['fingers_kinematics'][:1000, 8])
axs[3].set_title('Kinematics Velocity')

# Show the plot
plt.tight_layout()
plt.show()
print('done')