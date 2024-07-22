import numpy as np

# Dummy dataset similar to files_list processing
first_all_data_array = np.array([[1, 2, 3, 4, 5],
                                 [5, 6, 7, 8, 8],
                                 [1, 1, 3, 4, 5],
                                 [1, 2, 3, 4, 9]])
second_all_data_array = np.array([[1, 2, 3, 4, 5],
                                 [1, 2, 3, 4, 8],
                                 [1, 1, 3, 4, 5],
                                 [5, 6, 7, 8, 9]])
third_all_data_array = np.array([[1, 2, 3, 4, 5],
                                 [1, 2, 10, 4, 8],
                                 [5, 6, 7, 8, 5],
                                 [1, 2, 3, 4, 9]])

# Mimicking the get_dataset_data function
def get_dataset_data(dataset):
    return None, dataset, None, None

# Dummy variables to mimic the real scenario
first_processed_array_dict = None
first_dataset = "first"
files_list = ["first", "second", "third"]

# Your processing logic
datasets_array_list = []
datasets_array_sizes_list = []
for cd_id, current_data_set_filter in enumerate(files_list):
    if current_data_set_filter == first_dataset:
        current_processed_dict_filter, current_all_array_filter = first_processed_array_dict, first_all_data_array
    else:
        if current_data_set_filter == "second":
            current_all_array_filter = second_all_data_array
        elif current_data_set_filter == "third":
            current_all_array_filter = third_all_data_array
        current_processed_dict_filter, current_all_array_filter, _, _ = get_dataset_data(current_all_array_filter)

    datasets_array_sizes_list.append(len(current_all_array_filter))
    datasets_array_list.append(current_all_array_filter)

# Find the dataset with the smallest size (most constraining dataset)
most_constraining_dataset = first_all_data_array

filtered_datasets_list = []

# Filter each dataset in datasets_array_list
for dataset in datasets_array_list:
    # Find rows in the dataset where the first four elements match with any row in the most_constraining_dataset
    mask = np.isin(dataset[:, :4].view([('', dataset.dtype)]*4),
                   most_constraining_dataset[:, :4].view([('', most_constraining_dataset.dtype)]*4)).all(axis=1)
    filtered_dataset = dataset[mask]
    filtered_datasets_list.append(filtered_dataset)

# Display filtered datasets
for idx, filtered_dataset in enumerate(filtered_datasets_list):
    print(f"Filtered dataset size {idx}:\n{np.shape(filtered_dataset)}\n")
