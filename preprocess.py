import os
import librosa
import math
import json

DATASET_PATH = "C:/Users/casper/Desktop/genres/genres"
JSON_PATH = "C:/Users/casper/Desktop/data.json"
SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=30, n_fft=2048, hop_length=512, num_segment=300):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],  # training input
        "labels": []  # training output
    }
    # Müziğin süresi *
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segment)

    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  # 1.2 -> 2

    # current folder, dirnames of subfolders, filenames all the files in folders
    # we can recursively tüm dosyaları gezeceğiz. walk() metodu ile
    # ayrıca saymak istersek enumerate kullanabiliriz.
    # ensure that we're not at the root level
    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/")
            print("dirpath_components = ", dirpath_components)
            print("dirpath_components[-1] = ", dirpath_components[-1])
            semantic_label = dirpath_components[-1]
            print("semantic_label = 0", semantic_label)
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                print("file_path = ", file_path)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segment):
                    start_sample = num_samples_per_segment * s  # s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segment  # s=0 -> num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store mfcc for segment if it has expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print(f"{file_path}, segment:{s}")
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segment=5)
