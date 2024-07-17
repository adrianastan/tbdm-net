# tbdm-net
Official implementation of TBDM-Net

### Data preparation
MFCC extraction is done in a similar manner to [TIM-Net](https://github.com/Jiaxin-Ye/TIM-Net_SER).
For convenience, the MFCC extraction function is the following: (see TIM-Net's official repository for details):
```
def get_feature(signal, feature_type:str="MFCC", mean_signal_length:int=310000, embed_len: int = 39):
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
```
To create a dataset, extract MFCC for all your audio files, and perform one-hot encoding on the emotion labels.
Concatenate the MFCC into a single numpy array, and the labels into another array. Save them both into a dictionary
with keys 'x' for the features and 'y' for the labels. A short example of what the dictionary should look like when
saving to disk:
```
    d_train = {'x': train_mfcc, 'y': train_labels}
    with open('IEMOCAP_TRAIN.npy', 'wb') as f:
        np.save(f, d_train)
```

### Running the model
The file containing the full model is SER_v18.py. The ablation study models with partially-disabled functionality
can be found in the SER_v18_ablation...py files.
To run using one of the existing configurations(please note you will need to extract the features yourself
and place them in the same directory as SER_v18.py)
you can simply run:
```
python3 SER_v18.py
```
Configuring the training behaviour can be done through the following variables in the main file:
- `DATASET_NAME` for the dataset you want to choose (pre-made configurations are available for "iemocap",
"emodb", "casia", "emovo", "ravdess", and "meld")
- `K_FOLDS` for the number of cross validation folds
- `N_EPOCHS` for the number of epochs
- `BATCH_SIZE` for the batch size
- `RANDOM_SEED` for the random seed.
An example configuration:
```
DATASET_NAME = "iemocap"
K_FOLDS = 10
N_EPOCHS = 300
BATCH_SIZE = 64
RANDOM_SEED = 46
```
If you wish to alter the default save paths, you can change the path_results variable to a path of your choice.
To add a new dataset configuration, just add a new entry for your dataset name in the `paths_data`, `paths_all_best_uars`,
`paths_all_best_wars`, etc. dictionaries in the main file.

### Results
The results are saved to the path specified by the `path_results` variable. Model checkpoints are saved
to `paths_all_best_models_mid` and `paths_all_best_models_end` respectively. `paths_all_best_models_mid` contains
the models that have performed best on the training set during training, and `paths_all_best_models_end` contains
the models in their final state after training for `N_EPOCHS` epochs.


### Gender Experiments
For the pre-hoc methods we have presented, gender information must be concatenated to the MFCC before input.
Such an extraction script can be seen in `create_mfccs_prehoc.py`.
The gender classifier we have trained is readily available in `trained_gender_classifier_proba.pkl`.
It can be loaded and used as such(note the fact that it operates on nemo encodings):
```
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    model_name='titanet_large')
gender_clf = pickle.load(
    open('trained_gender_classifier_proba.pkl', 'rb'))
embs = speaker_model.get_embedding(audio_file_path).cpu().numpy()
predicted_gender = gender_clf.predict(embs)

```
After creating a dataset with these features concatenated to the input, simply run them through `SER_v18.py`.

The post-hoc benchmark script is located in `SER_v18_test_only.py` and relies on the model checkpoints
saved by `SER_v18.py`. When running these tests, make sure the random seed is the same
that was used during training, otherwise the dataset splits will be different,
rendering the experiment useless.

To compute metrics for one of the post-hoc methods, uncomment the associated code block. For example,
to perform the "Golden Labels" benchmark, uncomment this block of code from `SER_v18_test_only.py`:
```
    # GOLDEN LABELS
    # for batch in test_loader_male:
    #     X, Y, X_embs = batch
    #     X = X.float()
    #     Y = torch.argmax(Y, dim=1)
    #     y_local = curr_male(X).to(device)
    #     y_local = torch.argmax(y_local, dim=1)
    #     # print("y_local shape: ", y_local.shape)
    #     y_pred = torch.cat((y_pred, y_local))
    #     y_true = torch.cat((y_true, Y))
    # for batch in test_loader_female:
    #     X, Y, X_embs = batch
    #     X = X.float()
    #     Y = torch.argmax(Y, dim=1)
    #     y_local = curr_female(X).to(device)
    #     y_local = torch.argmax(y_local, dim=1)
    #     # print("y_local shape: ", y_local.shape)
    #     y_pred = torch.cat((y_pred, y_local))
    #     y_true = torch.cat((y_true, Y))
```


