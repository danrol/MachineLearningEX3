import librosa
import scipy.stats as stats

fpath = "test_files/00d4b9cd-3619-482a-9941-b5dc2b277cc5.wav"
y, sr = librosa.load(fpath, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
mfcc = stats.zscore(mfcc, axis=1) # Normalization

def main():
    pass

if __name__ == "__main__":
    main()
