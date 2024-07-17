import sys,os
import librosa
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import nemo.collections.asr as nemo_asr

emotion_map = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}

def get_feature(signal, feature_type:str="MFCC", mean_signal_length:int=310000, embed_len: int = 39):
#    feature = None
#    signal, fs = librosa.load(file_path)# Default setting on sampling rate
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    if feature_type == "MFCC":
        mfcc =  librosa.feature.mfcc(y=signal, sr=22050, n_mfcc=embed_len)
        feature = np.transpose(mfcc)
    return feature

def main():

    # feats = np.zeros((5531, 606, 41))
    # feats = np.zeros((5531))
    feats = np.zeros((5531, 606, 40))
    # feats = np.zeros((2882, 606, 39))
    # feats = np.zeros((2764, 606, 39))
    # feats = np.zeros((2649, 606, 39))
    paths = {}
    with open('wav_list_vlad.txt') as fin:
        for line in fin.readlines():
            paths[os.path.split(line.strip())[1][:-4]] = line.strip()
    i = 0
    labels = []
    embs = []
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_large')
    gender_clf = pickle.load(
            open('trained_gender_classifier_proba.pkl', 'rb'))

    with open('iemocap_metadata_processed.csv') as fin:
        for line in tqdm(fin.readlines()[1:]):
            # if i>=2764:
            #     break
            data = line.strip().split(',')
            start = float(data[1])
            end = float(data[2])
            f = data[3]
            gender_indicator = f[-4]
            if gender_indicator == 'F':
                gender = 0
            if gender_indicator == 'M':
                gender = 1
            # feats[i] = gender
            # i+=1
            # continue
            desired_gender = 0 # 0 FOR FEMALE, 1 FOR MALE
            # if gender == desired_gender:
            if True:

                wav, sr = librosa.load(paths[f])
                embs_local = speaker_model.get_embedding(paths[f]).cpu().numpy()
                ft = get_feature(wav)
                # pg_proba = gender_clf.predict_proba(embs_local)
                # pg_proba = np.array(pg_proba)
                pg = gender_clf.predict(embs_local)
                pg = np.array(pg).reshape(-1, 1)

                n_rep = ft.shape[0]
                # pg_proba = np.repeat(pg_proba, n_rep, axis=0)
                pg = np.repeat(pg, n_rep, axis=0)
                # golden = np.array([[gender]])
                # golden = np.repeat(golden, n_rep, axis=0)
                # ft = np.concatenate((ft, pg_proba), axis=1)
                ft = np.concatenate((ft, pg), axis=1)
                # ft = np.concatenate((ft, golden), axis=1)
    #            print(ft)
                feats[i,:,:] = ft
                i+=1
                emotion_1h = [0,0,0,0]
                emotion_1h[emotion_map[data[4]]] = 1
                labels.append(deepcopy(emotion_1h))
                embs.append(deepcopy(embs_local))

    d = {'x':feats, 'y':np.array(labels), 'x_embs':np.array(embs)}

    with open('IEMOCAP_ADRIANA_VLAD_CONCAT_PRED_BINARY_FULL_WITH_EMBS.npy', 'wb') as f:
        np.save(f, d)
    # np.save('IEMOCAP_ADRIANA_VLAD_GENDERS.npy', feats)

if __name__ =='__main__':
    main()
