from neuraldecoding.dataset import Dataset

subject = 'Joker'
date = '2024-08-23'
runs = [1,2]

dataset = Dataset()

# set the server directory to a custom path - not necessary if the data is in the default path (Z:/ for Windows systems)
# dataset.set_server_directory('F:')

# loading data using the subject-date-runs format
dataset.load_data(subject_name=subject, date=date, runs=runs)

# loading data using the data_path format
# data_path = 'Z:\\Data\\Monkeys\\Joker\\2024-08-23'
# dataset.load_data(data_path=data_path, runs=runs)

print('Loading data')