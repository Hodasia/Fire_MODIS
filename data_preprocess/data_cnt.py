import numpy as np
import os
def count_files(directory_path):
    try:
        # List all files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        
        # Print the count
        print(f"Number of files in '{directory_path}': {len(files)}")
        
        # print the list of files
        # print("List of files:")
        # for file in files:
        #     print(file)
    except FileNotFoundError:
        print(f"The directory '{directory_path}' does not exist.")
    except PermissionError:
        print(f"Permission error accessing '{directory_path}'.")

path_parent = "/workplace/dataset/MODIS_new/"

path_2021 = os.path.join(path_parent, '2021_new')
path_2022 = os.path.join(path_parent, '2022_new')

path_pos = os.path.join(path_parent, 'train_new', 'positive')
path_neg = os.path.join(path_parent, 'train_new', 'negative')
path_pos_80 = os.path.join(path_parent, 'train_80', 'positive')
path_neg_80 = os.path.join(path_parent, 'train_80', 'negative')
path_pos_60 = os.path.join(path_parent, 'train_60', 'positive')
path_neg_60 = os.path.join(path_parent, 'train_60', 'negative')

path_valid = os.path.join(path_parent, 'valid_new')
path_valid_80 = os.path.join(path_parent, 'valid_80')
path_test = os.path.join(path_parent, 'test_new')
path_test_80 = os.path.join(path_parent, 'test_80')


# count_files(path_2021) # 2585 (11, 2030, 1354)
# count_files(path_2022) # 2148 (11, 2030, 1354)

#############################################
## Con_thrd=60 old
# count_files(path_pos) # 10214 (64, 11, 15, 15)
# count_files(path_neg) # 64097 (64, 11, 15, 15)
# count_files(path_valid) # 1074
# count_files(path_test) # 1074

#############################################
## Con_thrd=80
# count_files(path_pos_80) # 4843 (64, 11, 15, 15)
# count_files(path_neg_80) # 8335 (64, 11, 15, 15)
# count_files(path_valid_80) # 1074
# count_files(path_test_80) # 1074

#############################################
## Con_thrd=60 new
# count_files(path_pos_60) # 10214 (64, 11, 15, 15)
# count_files(path_neg_60) # 20410 (64, 11, 15, 15)

# pos_sample = np.load(os.path.join(path_valid_80, '1074.npy')) # (64, 11, 15, 15)
# print(pos_sample.shape)
