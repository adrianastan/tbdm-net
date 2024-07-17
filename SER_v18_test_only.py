import numpy as np
import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
import dill
import pickle
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torchmetrics import Precision, Recall, F1Score
import gc
from torch.utils.data import TensorDataset
import torchvision

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

def get_metrics(model, test_loader):
    uar_f = Recall(task="multiclass", average="macro", num_classes=CLASSES).to(device)
    war_f = Recall(task="multiclass", average="weighted", num_classes=CLASSES).to(device)
    f1_f = F1Score(task="multiclass", average="weighted", num_classes=CLASSES).to(device)

    with torch.no_grad():
        model.eval()

        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)

        for batch in test_loader:
            X, y, _ = batch
            X = X.float()
            
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))

        uar = uar_f(Y, Y_hat)
        war = war_f(Y, Y_hat)
        f1 = f1_f(Y, Y_hat)
    return uar, war, f1


with open('all_best_models_iemocap_end.pkl', "rb") as f:
    ALL_BEST_MODELS = dill.load(f)

RANDOM_SEED = 42
K_FOLDS = 10
BATCH_SIZE = 64

gc.collect()

data = np.load("./IEMOCAP_ADRIANA_VLAD_FULL_WITH_EMBS.npy", allow_pickle=True).item()
genders = np.load("./IEMOCAP_ADRIANA_VLAD_GENDERS.npy", allow_pickle=True)

X = data["x"].transpose(0, 2, 1)
X_embs = data["x_embs"].squeeze()
Y = data["y"]
Y_1h = np.zeros((Y.shape[0], 4))
for i, y in enumerate(Y):
    Y_1h[i][y]=1
X = torch.tensor(X, device=device, dtype=torch.float64)
X_embs = torch.tensor(X_embs, device=device, dtype=torch.float64)
Y_1h = torch.tensor(Y_1h, device=device, dtype=torch.float64)
print(Y)
print("X shape: ", X.shape, " Y shape: ", Y_1h.shape, " X_embs shape: ", X_embs.shape)
CLASSES = Y_1h.shape[1]
print('CLASSES: ', CLASSES)


dataset = TensorDataset(X, Y_1h, X_embs)
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
ALL_UAR = []
ALL_WAR = []
ALL_F1 = []

test_ds_male_splits = []
test_ds_female_splits = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset), 1):
    male_idx = [i for i in test_idx if genders[i]==1]
    female_idx = [i for i in test_idx if genders[i]==0]
    test_ds_male_splits.append(deepcopy(male_idx))
    test_ds_female_splits.append(deepcopy(female_idx))


for fold, (male_split, female_split) in enumerate(zip(test_ds_male_splits, test_ds_female_splits), 1):
    test_sampler_male = torch.utils.data.SubsetRandomSampler(male_split)
    test_sampler_female = torch.utils.data.SubsetRandomSampler(female_split)

    test_loader_male = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler_male
    )
    test_loader_female = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler_female
    )


    curr_male = ALL_BEST_MODELS[fold-1]
    curr_female = ALL_BEST_MODELS[fold-1]
    # uar, war, f1 = get_metrics(curr_male, test_loader_male)
    uar, war, f1 = get_metrics(curr_female, test_loader_female)
    print(uar, war, f1)
    ALL_UAR.append(uar.cpu().numpy())
    ALL_WAR.append(war.cpu().numpy())
    ALL_F1.append(f1.cpu().numpy())
    continue


    curr_male.eval()
    curr_female.eval()


    y_true = torch.tensor([], device=device)
    y_pred = torch.tensor([], device=device)


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

    # PREDICTION BINARY
    # for batch in test_loader_male:
    #     a,b,c = batch
    #     for X, Y, X_embs in zip(a,b,c):
    #         # print("---------", X_embs.shape)
    #         X_embs = X_embs.unsqueeze(0)
    #         X = X.unsqueeze(0)
    #         X = X.float()
    #         predicted_gender = gender_clf.predict(X_embs.cpu().numpy())
    #         Y = torch.argmax(Y)
    #         Y = Y.unsqueeze(0)
    #         if predicted_gender == 0:
    #             y_local = curr_male(X).to(device)
    #         else:
    #             y_local = curr_female(X).to(device)
    #         y_local = torch.argmax(y_local, dim=1)
    #         # print("y_local shape: ", y_local.shape)
    #         y_pred = torch.cat((y_pred, y_local))
    #         y_true = torch.cat((y_true, Y))
    # for batch in test_loader_female:
    #     a,b,c = batch
    #     for X, Y, X_embs in zip(a,b,c):
    #         X_embs = X_embs.unsqueeze(0)
    #         X = X.unsqueeze(0)
    #         X = X.float()
    #         predicted_gender = gender_clf.predict(X_embs.cpu().numpy())
    #         Y = torch.argmax(Y)
    #         Y = Y.unsqueeze(0)
    #         if predicted_gender == 0:
    #             y_local = curr_male(X).to(device)
    #         else:
    #             y_local = curr_female(X).to(device)
    #         y_local = torch.argmax(y_local, dim=1)
    #         # print("y_local shape: ", y_local.shape)
    #         y_pred = torch.cat((y_pred, y_local))
    #         y_true = torch.cat((y_true, Y))

    # PREDICTION W/ PROBS
    # for batch in test_loader_male:
    #     a,b,c = batch
    #     for X, Y, X_embs in zip(a,b,c):
    #         # print('X shape: ', X.shape)
    #         X_embs = X_embs.unsqueeze(0)
    #         X = X.unsqueeze(0)
    #         X = X.float()
    #         pg_proba = gender_clf.predict_proba(X_embs.cpu().numpy())
    #         m_p = float(pg_proba[0][0])
    #         f_p = float(pg_proba[0][1])
    #         # print(pg_proba)
    #         # print(m_p, f_p)
    #         Y = torch.argmax(Y)
    #         Y = Y.unsqueeze(0)
    #         y_local_male = curr_male(X).to(device)
    #         y_local_female = curr_female(X).to(device)
    #         y_local = y_local_male*m_p + y_local_female*f_p
    #         y_local = torch.argmax(y_local, dim=1)
    #         # print("y_local shape: ", y_local.shape)
    #         y_pred = torch.cat((y_pred, y_local))
    #         y_true = torch.cat((y_true, Y))
    # for batch in test_loader_female:
    #     a,b,c = batch
    #     for X, Y, X_embs in zip(a,b,c):
    #         X_embs = X_embs.unsqueeze(0)
    #         X = X.unsqueeze(0)
    #         X = X.float()
    #         # predicted_gender = gender_clf.predict(X_embs.cpu().numpy())
    #         pg_proba = gender_clf.predict_proba(X_embs.cpu().numpy())
    #         m_p = float(pg_proba[0][0])
    #         f_p = float(pg_proba[0][1])
    #         # print(pg_proba)
    #         # print(m_p, f_p)
    #         Y = torch.argmax(Y)
    #         Y = Y.unsqueeze(0)
    #         y_local_male = curr_male(X).to(device)
    #         y_local_female = curr_female(X).to(device)
    #         y_local = y_local_male*m_p + y_local_female*f_p
    #         y_local = torch.argmax(y_local, dim=1)
    #         # print("y_local shape: ", y_local.shape)
    #         y_pred = torch.cat((y_pred, y_local))
    #         y_true = torch.cat((y_true, Y))

    uar_f = Recall(task="multiclass", average="macro", num_classes=CLASSES).to(device)
    war_f = Recall(task="multiclass", average="weighted", num_classes=CLASSES).to(device)
    f1_f = F1Score(task="multiclass", average="weighted", num_classes=CLASSES).to(device)
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    print(y_true.shape, y_pred.shape)
    # print(y_true)
    # print(y_pred)
    uar = uar_f(y_true, y_pred)
    war = war_f(y_true, y_pred)
    f1 = f1_f(y_true, y_pred)
    print("UAR: ", uar)
    print("WAR: ", war)
    print("F1: ", f1)
    ALL_UAR.append(uar.cpu().numpy())
    ALL_WAR.append(war.cpu().numpy())
    ALL_F1.append(f1.cpu().numpy())

# IDX = np.array(IDX)
# with open('IDX.npy', 'rb') as f:
#     OLD_IDX = np.load(f)
# print(np.array_equal(IDX, OLD_IDX))

print("ALL UAR: ", ALL_UAR)
print("ALL WAR: ", ALL_WAR)
print("ALL F1: ", ALL_F1)
print("-------------------")
print("AVG UAR: ", np.mean(np.array(ALL_UAR)))
print("AVG WAR: ", np.mean(np.array(ALL_WAR)))
print("AVG F1: ", np.mean(np.array(ALL_F1)))

            

    



