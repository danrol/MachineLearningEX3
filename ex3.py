import librosa
import scipy.stats as stats
import os
import numpy as np
from scipy.ndimage.interpolation import shift


### 1. from test wav file take mfcc
### 2. create 20 points (number of feautures) with 32 dimensions (number of feautures)
### 3. feed new non test wav file
### 4. using knn find closest number. possible normal(mfcc1 - mfcc2)
### 5. print results
# train_one_path = "train_data/one"
# train_two_path = "train_data/two"
# train_three_path = "train_data/three"
# train_four_path = "train_data/four"
# train_five_path = "train_data/five"
huge_num = 100000
train_data_paths = ["train_data/one", "train_data/two", "train_data/three", "train_data/four", "train_data/five"]
test_fpath = "test_files"
# y, sr = librosa.load(fpath, sr=None)
# mfcc = librosa.feature.mfcc(y=y, sr=sr)
# mfcc = stats.zscore(mfcc, axis=1) # Normalization
k = 1
all_mfccs = {"1": [], "2": [], "3": [], "4": [], "5": []}


def init_min_distances():
    return np.full((2, k), huge_num)


def check_min_distance(min_distances, new_distance, train_point_index):
    min_distances[min_distances[:,1].argsort()]
    for index, distance in enumerate(min_distances):
        if new_distance < distance:
            shift(min_distances, 1, cval=np.NaN)
            min_distances[index] = np.array([train_point_index, new_distance])
            break
    return min_distances

def find_knn(mfcc_to_check):
    all_distances = np.array()
    min_distances = init_min_distances()
    for index, mfccs in all_mfccs.items():
        for mfcc in mfccs:
            distance = np.linalg.norm(mfcc - mfcc_to_check)
            min_distances = check_min_distance(min_distances, distance, index)
    return min_distances


def check_test_file():
    for filename in os.listdir(test_fpath):
        if filename.endswith(".wav"):
            mfcc_to_check = get_mfcc(test_fpath+"/"+filename)
            find_min_distance(mfcc_to_check)
            #all_mfccs[str(index)].extend(get_mfcc(directory+"/"+filename))
def predict(fpath):
    pass
    # print(mfcc)


def print_all_mfccs():
    for i in all_mfccs:
        print(f"****** mfccs for number {i} ******:\n{all_mfccs[i]}\n\n")


def get_mfcc(fpath):
    y, sr = librosa.load(fpath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = stats.zscore(mfcc, axis=1)  # Normalization
    return mfcc


def save_all_mfccs_from_train_data():
    for (index,directory) in enumerate(train_data_paths, start=1):
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                all_mfccs[str(index)].extend(get_mfcc(directory+"/"+filename))


def main():
    save_all_mfccs_from_train_data()
    print_all_mfccs()

if __name__ == "__main__":
    main()
