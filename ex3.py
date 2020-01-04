import librosa
import scipy.stats as stats
import os
import numpy as np
from scipy.ndimage.interpolation import shift

num_of_numbers_to_check = 5
huge_num = 100000
train_data_paths = ["train_data/one", "train_data/two", "train_data/three", "train_data/four", "train_data/five"]
test_fpath = "test_files"
#test_fpath = "train_data_proper_file_names"
output_file_path = "output.txt"
k = 1
all_mfccs = {"1": [], "2": [], "3": [], "4": [], "5": []} #all mfccs from train data divided by label
results = [] # final results (filename, predicted label')


# init matrix of of distance. every row [train_point_label, distance]. number of rows equal to k
def init_min_distances():
    return np.full((k, 2), huge_num, dtype=float)


def check_min_distance(min_distances, new_distance, train_point_index):
    min_distances = min_distances[min_distances[:, 1].argsort()]
    for index, distance in enumerate(min_distances):
        if new_distance < distance[1]:
            shift(min_distances, 1, cval=np.NaN)
            min_distances[index] = np.array([train_point_index, new_distance])
            break
    return min_distances


# find knn for wav
def find_knn(mfcc_to_check):
    min_distances = init_min_distances()
    for index, mfccs in all_mfccs.items():
        for mfcc in mfccs:
            distance = np.linalg.norm(mfcc - mfcc_to_check)
            min_distances = check_min_distance(min_distances, distance, index)
    return min_distances


# method returns final prediction for wav
def final_prediction(min_distances_for_test_file):
    occurances_counter = [0] * num_of_numbers_to_check
    for distance in min_distances_for_test_file:
        counter_index = int(distance[0]) - 1
        # if distance[1] != huge_num:
        occurances_counter[counter_index] = occurances_counter[counter_index] + 1
    return np.argmax(occurances_counter) + 1

# find predictions for test files
def check_test_files():
    for filename in os.listdir(test_fpath):
        if filename.endswith(".wav"):
            mfcc_to_check = get_mfcc(test_fpath+"/"+filename)
            min_distances_for_test_file = find_knn(mfcc_to_check)
            prediction = final_prediction(min_distances_for_test_file)
            results.append((filename, prediction))


def print_results():
    print(f"****** Results k = {k}******")
    for result in results:
        print(result)


def write_output_to_file():
    file = open(output_file_path, 'w')
    for result in results:
        file.write(str(result[0]) + " - "+ str(result[1])+"\n")
    file.close()


def print_all_mfccs():
    for i in all_mfccs:
        print(f"****** mfccs for number {i} ******:\n{all_mfccs[i]}\n\n")


def get_mfcc(fpath):
    y, sr = librosa.load(fpath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = stats.zscore(mfcc, axis=1)  # Normalization
    return mfcc

# get all mfccs from train data and save them to all_mfccs
def save_all_mfccs_from_train_data():
    for (index, directory) in enumerate(train_data_paths, start=1):
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                all_mfccs[str(index)].extend(get_mfcc(directory+"/"+filename))


def main():
    save_all_mfccs_from_train_data()
    # print_all_mfccs()
    check_test_files()
    print_results()
    write_output_to_file()


if __name__ == "__main__":
    main()