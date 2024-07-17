import numpy as np
import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
import dill
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torchmetrics import Precision, Recall, F1Score
import gc
from torch.utils.data import TensorDataset
import torchvision

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
# # Model


class Temporal_Aware_Block(nn.Module):
    def __init__(
        self, nb_filters, kernel_size, lin_size, dropout_rate=0.1, dilation_rate=1
    ):
        super(Temporal_Aware_Block, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv_1_1 = nn.Conv1d(
            nb_filters,
            nb_filters,
            kernel_size,
            dilation=dilation_rate,
            padding="same",
            bias=False,
        )
        self.bn_1_1 = nn.BatchNorm1d(nb_filters)
        self.activation = nn.GELU()

        self.conv_2_1 = nn.Conv1d(
            nb_filters,
            nb_filters,
            kernel_size,
            dilation=dilation_rate,
            padding="same",
            bias=False,
        )
        self.bn_2_1 = nn.BatchNorm1d(nb_filters)
        self.activation2 = nn.GELU()

        self.spatial_dropout = nn.Dropout2d(p=self.dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.lin = nn.Linear(lin_size, lin_size)

    def forward(self, x):
        original_x = x
        # 1.1
        x = self.conv_1_1(x)
        x = self.bn_1_1(x)
        x = self.activation(x)

        # spatial dropout simulation
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]

        # 2.1
        x = self.conv_2_1(x)
        x = self.bn_2_1(x)
        x = self.activation2(x)
        # spatial dropout simulation
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]

        x = self.sigmoid(x)
        F_x = torch.mul(original_x, x)
        F_x = torch.cat([F_x, original_x], dim=1)
        return F_x


class WeightLayer(nn.Module):
    def __init__(self, dim):
        super(WeightLayer, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(dim, 1).uniform_())

    def forward(self, x):
        # print("x.shape: ", x.shape)
        # print("self.kernel.shape: ", self.kernel.shape)
        tempx = x.transpose(1, 2)
        x = torch.matmul(tempx, self.kernel)
        x = x.squeeze(-1)
        return x


class Dimension_Change(nn.Module):
    def __init__(self, in_size, out_size):
        super(Dimension_Change, self).__init__()
        self.conv = nn.Conv1d(in_size, out_size, 1, padding="same", bias=False)
        self.bn = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ### TBDMNet


class TBDMNet(nn.Module):
    def __init__(
        self,
        classes,
        sequence_length,
        nb_features,
        kernel_size=2,
        nb_stacks=1,
        dilations=None,
        activation="relu",
        dropout_rate=0.1,
        return_sequences=True,
        name="TIMNET",
    ):
        super(TBDMNet, self).__init__()

        self.classes = classes
        self.sequence_length = sequence_length
        self.nb_features = nb_features

        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size

        self.supports_masking = True
        self.mask_value = 0.0
        self.multiplier = 1
        self.nr_blocks = 6

        if dilations is None:
            self.dilations = [2**i for i in range(8)]

        self.dilations_1 = self.dilations[: self.nr_blocks]

        self.tan_layers_fwd_1 = nn.ModuleList()
        for s in range(nb_stacks):
            for i, d in enumerate(self.dilations_1):
                multiplier = 2 ** (i)
                layer = Temporal_Aware_Block(
                    nb_features * multiplier,
                    kernel_size,
                    SEQUENCE_LENGTH,
                    dropout_rate,
                    dilation_rate=d,
                )
                self.tan_layers_fwd_1.append(layer)
                self.multiplier = multiplier

        self.tan_layers_bkwd_1 = nn.ModuleList()
        for s in range(nb_stacks):
            for i, d in enumerate(self.dilations_1):
                multiplier = 2 ** (i)
                layer = Temporal_Aware_Block(
                    nb_features * multiplier,
                    kernel_size,
                    sequence_length,
                    dropout_rate,
                    dilation_rate=d,
                )
                self.tan_layers_bkwd_1.append(layer)
                self.multiplier = multiplier

        self.dimensionality_reduction_layers = nn.ModuleList()
        for s in range(nb_stacks):
            for i, d in enumerate(self.dilations_1):
                multiplier = 2 ** (i + 1)
                self.dimensionality_reduction_layers.append(
                    Dimension_Change(nb_features * multiplier, nb_features)
                )

        self.weight_layer = WeightLayer(self.nr_blocks)
        self.fc = nn.Linear(self.nb_features, self.classes)
        self.softmax = nn.Softmax(-1)
        self.avgpool = torch.nn.AvgPool1d(self.sequence_length)

    def forward(self, inputs):
        forward = inputs
        backward = torch.flip(inputs, dims=[2])
        final_skip_connection = []
        skip_out_forward = forward
        skip_out_backward = backward

        for s in range(self.nb_stacks):
            for d in range(len(self.dilations_1)):
                skip_out_forward = self.tan_layers_fwd_1[d](skip_out_forward)
                skip_out_backward = self.tan_layers_bkwd_1[d](skip_out_backward)
                temp_skip = skip_out_forward + skip_out_backward
                temp_skip = self.dimensionality_reduction_layers[d](temp_skip)
                temp_skip = self.avgpool(temp_skip)
                final_skip_connection.append(temp_skip)

        output_2 = final_skip_connection[0]
        for i, item in enumerate(final_skip_connection):
            if i == 0:
                continue
            output_2 = torch.cat([output_2, item], dim=-1)
        x = output_2.transpose(2, 1)
        x = self.weight_layer(x)
        x = self.fc(x)
        return x


# ## Training function


def train(
    num_epochs,
    model,
    best_model,
    optimizer,
    loss_fn,
    train_loader,
    test_loader,
    classes,
    k,
    writer,
):
    all_losses = []
    uar_f = Recall(task="multiclass", average="macro", num_classes=CLASSES).to(device)
    war_f = Recall(task="multiclass", average="weighted", num_classes=CLASSES).to(
        device
    )
    f1_f = F1Score(task="multiclass", num_classes=CLASSES).to(device)
    MAX_VAL_UAR = 0
    MAX_VAL_WAR = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_iters = 0

        model.train()
        Y = torch.Tensor().to(device)
        Y_hat = torch.Tensor().to(device)
        loop = iter(train_loader)
        for batch in loop:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            optimizer.zero_grad()

            y_hat = model(X)
            y_hat_class = torch.argmax(y_hat, dim=1)

            Y_hat = torch.cat((Y_hat, y_hat_class))

            loss = loss_fn(y_hat, y)
            # loss = loss_fn(y_hat, y, reduction="sum")
            all_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_iters += 1

        # METRICS
        with torch.no_grad():
            model.eval()
            Y_test = torch.tensor([], device=device)
            Y_hat_test = torch.tensor([], device=device)
            for batch in test_loader:
                X, y = batch
                y_class = torch.argmax(y, dim=1)
                Y_test = torch.cat((Y_test, y_class))
                y_hat = model(X)
                y_hat_class = torch.argmax(y_hat, dim=1)
                Y_hat_test = torch.cat((Y_hat_test, y_hat_class))
            uar_test = uar_f(Y_test, Y_hat_test)
            war_test = war_f(Y_test, Y_hat_test)
            f1_test = f1_f(Y_test, Y_hat_test)
            writer.add_scalar("UAR/test k=" + str(k), uar_test.cpu().detach(), epoch)
            writer.add_scalar("WAR/test k=" + str(k), war_test.cpu().detach(), epoch)
            writer.add_scalar("F1/test k=" + str(k), f1_test.cpu().detach(), epoch)

            Y_train = torch.tensor([], device=device)
            Y_hat_train = torch.tensor([], device=device)
            for batch in train_loader:
                X, y = batch
                y_class = torch.argmax(y, dim=1)
                Y_train = torch.cat((Y_train, y_class))
                y_hat = model(X)
                y_hat_class = torch.argmax(y_hat, dim=1)
                Y_hat_train = torch.cat((Y_hat_train, y_hat_class))
            uar_train = uar_f(Y_train, Y_hat_train)
            war_train = war_f(Y_train, Y_hat_train)
            f1_train = f1_f(Y_train, Y_hat_train)
            writer.add_scalar("UAR/train k=" + str(k), uar_train.cpu().detach(), epoch)
            writer.add_scalar("WAR/train k=" + str(k), war_train.cpu().detach(), epoch)
            writer.add_scalar("F1/train k=" + str(k), f1_test.cpu().detach(), epoch)
            writer.flush()
            if uar_train > MAX_VAL_UAR:
                MAX_VAL_UAR = uar_train

            if war_train > MAX_VAL_WAR:
                MAX_VAL_WAR = war_train
                best_model = deepcopy(model)

    return best_model, MAX_VAL_UAR, MAX_VAL_WAR, all_losses

def get_metrics(model, test_loader):
    uar_f = Recall(task="multiclass", average="macro", num_classes=CLASSES).to(device)
    war_f = Recall(task="multiclass", average="weighted", num_classes=CLASSES).to(device)
    
        
    f1_f = F1Score(task="multiclass", average="weighted", num_classes=CLASSES).to(device)
    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        for batch in test_loader:
            X, y = batch
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))
            y_hat = model(X)
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
        uar = uar_f(Y, Y_hat)
        war = war_f(Y, Y_hat)
        f1 = f1_f(Y, Y_hat)

    return uar, war, f1


# # Dataset switcher


paths_data = {
    "casia": "./CASIA.npy",
    "emovo": "./EMOVO.npy",
    "ravdess": "./RAVDE.npy",
    # "ravdess_exp": "./RAVDESS_HALF.npy",
    # "ravdess_exp": "./RAVDESS_MALE.npy",
    "ravdess_exp": "./RAVDESS_FEMALE.npy",
    # "ravdess_exp": "./RAVDESS_FULL.npy",
    "emodb": "./EMODB.npy",
    "iemocap": "./IEMOCAP.npy",
    "savee": "./SAVEE.npy",
    "meld": "./MELD.npy",
}

paths_all_best_uars = {
    "casia": "./all_best_uars_casia.pkl",
    "emovo": "./all_best_uars_emovo.pkl",
    "ravdess": "./all_best_uars_ravdess.pkl",
    "ravdess_exp": "./all_best_uars_ravdess_exp.pkl",
    "emodb": "./all_best_uars_emodb.pkl",
    "iemocap": "./all_best_uars_iemocap.pkl",
    "savee": "./all_best_uars_savee.pkl",
    "meld": "./all_best_uars_meld.pkl",
}

paths_all_best_wars = {
    "casia": "./all_best_wars_casia.pkl",
    "emovo": "./all_best_wars_emovo.pkl",
    "ravdess": "./all_best_wars_ravdess.pkl",
    "ravdess_exp": "./all_best_wars_ravdess_exp.pkl",
    "emodb": "./all_best_wars_emodb.pkl",
    "iemocap": "./all_best_wars_iemocap.pkl",
    "savee": "./all_best_wars_savee.pkl",
    "meld": "./all_best_wars_meld.pkl",
}

paths_all_best_models = {
    "casia": "./all_best_models_casia.pkl",
    "emovo": "./all_best_models_emovo.pkl",
    "ravdess": "./all_best_models_ravdess.pkl",
    "ravdess_exp": "./all_best_models_ravdess_exp.pkl",
    "emodb": "./all_best_models_emodb.pkl",
    "iemocap": "./all_best_models_iemocap.pkl",
    "savee": "./all_best_models_savee.pkl",
    "meld": "./all_best_models.pkl",
}

paths_results = {
    "casia": "./results_casia.txt",
    "emovo": "./results_emovo.txt",
    "ravdess": "./results_ravdess.txt",
    "ravdess_exp": "./results_ravdess_exp.txt",
    "emodb": "./results_emodb.txt",
    "iemocap": "./results_iemocap.txt",
    "savee": "./results_savee.txt",
    "meld": "./results_meld.txt",
}

emotion_sets = {
    "casia": ["angry", "fear", "happy", "neutral", "sad", "surprise"],
    "emovo": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "ravdess": [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    ],

    "ravdess_exp": [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    ],
    "emodb": ["angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"],
    "iemocap": ["angry", "happy", "neutral", "sad"],
    "savee": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "meld": [
        "anger",
        "disgust",
        "sadness",
        "joy",
        "neutral",
        "surprise",
        "fear",
    ],
}

# ## Choose dataset ↓


DATASET_NAME = "ravdess_exp"
K_FOLDS = 10
N_EPOCHS = 300
BATCH_SIZE = 64
RANDOM_SEED = 46

path_data = paths_data[DATASET_NAME]
path_all_best_uars = paths_all_best_uars[DATASET_NAME]
path_all_best_wars = paths_all_best_wars[DATASET_NAME]
path_all_best_models = paths_all_best_models[DATASET_NAME]
path_results = paths_results[DATASET_NAME]
emotions = emotion_sets[DATASET_NAME]

gc.collect()

data = np.load(path_data, allow_pickle=True).item()
X = data["x"].transpose(0, 2, 1)
Y = data["y"]
X = torch.tensor(X, device=device)
Y = torch.tensor(Y, device=device, dtype=torch.float64)



print("X shape: ", X.shape, " Y shape: ", Y.shape)

dataset = TensorDataset(X, Y)
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

SEQUENCE_LENGTH = X.shape[-1]
CLASSES = len(Y[0])
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
loss_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_focal = torchvision.ops.focal_loss.sigmoid_focal_loss


# ## Train


warnings.filterwarnings("ignore")

gc.collect()

writer = SummaryWriter()
ALL_BEST_UARS_MID = []
ALL_BEST_WARS_MID = []
ALL_BEST_MODELS_MID = []
ALL_BEST_F1_MID = []
ALL_BEST_UARS_END = []
ALL_BEST_WARS_END = []
ALL_BEST_MODELS_END = []
ALL_BEST_F1_END = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset), 1):
    # train_ds = [dataset[i] for i in train_idx]
    test_ds_whole = [dataset[i] for i in test_idx]
    val_ds = deepcopy(test_ds_whole[:len(test_ds_whole)//5])
    test_ds = deepcopy(test_ds_whole[len(test_ds_whole)//5:])
     
    # Y = [t[1] for t in train_ds]
    # print(len(Y))
    # weight_count = {k: 0 for k in range(len(Y[0])) }
    # for y in Y:
    #     weight_count[torch.argmax(y).item()] +=1
    # print(weight_count)

    # class_weights = [1 / c for c in weight_count.values()]
    # print(class_weights)

    # sample_weights = [class_weights[torch.argmax(y).item()] for y in Y]

    # print(len(dataset))
    # print(len(train_ds))

    print(f"FOLD {fold}")
    print("--------------------------------")
    model = TBDMNet(
        nb_features=39, classes=CLASSES, sequence_length=SEQUENCE_LENGTH
    ).to(device)
    best_model = model
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.93, 0.98), lr=0.001)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    # train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    # drop_last when 1 item is left in the batch => error because mean for batchNorm cannot be calculated
    train_loader = DataLoader(
        # dataset=train_ds,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        # drop_last=True, # if the last batch is not full, drop it
    )
    test_loader = DataLoader(
        # dataset=train_ds,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        # drop_last=True, # if the last batch is not full, drop it
    )
    print(f"Total train samples: {len(train_idx)}")

    #validation_loader = DataLoader(
    #    dataset=val_ds,
    #    batch_size=BATCH_SIZE,
    #    # drop_last=True, # if the last batch is not full, drop it
    #)
    #test_loader = DataLoader(
    #    dataset=test_ds,
    #    batch_size=BATCH_SIZE,
    #    # drop_last=True, # if the last batch is not full, drop it
    #)
    print(f"Total test samples: {len(test_idx)}")
    # max_prec = 0
    # precision = Precision(average="macro", num_classes=CLASSES, task="multiclass").to(
        # device
    # )
    best_model, uar, war, all_losses = train(
        num_epochs=N_EPOCHS,
        model=model,
        best_model=best_model,
        optimizer=optimizer,
        loss_fn=loss_ce,
        # loss_fn=loss_focal,
        train_loader=train_loader,
        test_loader=test_loader,
        classes=CLASSES,
        k=fold,
        writer=writer,
    )

    # print("UAR(train):", uar)
    # print("WAR(train):", war)

    uar_mid, war_mid, f1_mid = get_metrics(best_model, test_loader)
    uar_end, war_end, f1_end = get_metrics(model, test_loader)

    print("UAR(test, best model on train):", uar_mid)
    print("WAR(test, best model on train):", war_mid)
    print("F1(test, best model on train):", f1_mid)
    print("---")
    print("UAR(last model):", uar_end)
    print("WAR(last model):", war_end)
    print("F1(last model):", f1_end)

    writer.add_scalar("Best UARs (mid)", uar_mid.cpu().detach(), fold)
    writer.add_scalar("Best WARs (mid)", war_mid.cpu().detach(), fold)
    writer.add_scalar("Best F1s (mid)", f1_mid.cpu().detach(), fold)
    writer.add_scalar("Best UARs (end)", uar_end.cpu().detach(), fold)
    writer.add_scalar("Best WARs (end)", war_end.cpu().detach(), fold)
    writer.add_scalar("Best F1s (end)", f1_end.cpu().detach(), fold)
    writer.flush()
    ALL_BEST_UARS_MID.append(uar_mid.cpu().detach())
    ALL_BEST_WARS_MID.append(war_mid.cpu().detach())
    ALL_BEST_F1_MID.append(f1_mid.cpu().detach())
    ALL_BEST_MODELS_MID.append(best_model)
    ALL_BEST_UARS_END.append(uar_end.cpu().detach())
    ALL_BEST_WARS_END.append(war_end.cpu().detach())
    ALL_BEST_F1_END.append(f1_end.cpu().detach())
    ALL_BEST_MODELS_END.append(deepcopy(model))
    # with open(path_all_best_uars, "wb") as f:
    #     dill.dump(ALL_BEST_UARS, f)
    # with open(path_all_best_wars, "wb") as f:
    #     dill.dump(ALL_BEST_WARS, f)
    # with open(path_all_best_models, "wb") as f:
    #     dill.dump(ALL_BEST_MODELS, f)

    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    gc.collect()

# ## Save Results


with torch.cuda.device(device):
    torch.cuda.empty_cache()
gc.collect()

# with open(path_all_best_uars, "rb") as f:
#     ALL_BEST_UARS = dill.load(f)
# with open(path_all_best_wars, "rb") as f:
#     ALL_BEST_WARS = dill.load(f)
# with open(path_all_best_models, "rb") as f:
#     ALL_BEST_MODELS = dill.load(f)


with open(path_results, "w") as f:
    f.write("ALL_BEST_UARs MID\n")
    f.write(str(ALL_BEST_UARS_MID) + "\n")
    f.write("ALL_BEST_WARs MID\n")
    f.write(str(ALL_BEST_WARS_MID) + "\n")
    f.write("ALL_BEST_F1s MID\n")
    f.write(str(ALL_BEST_F1_MID) + "\n")
    f.write("ALL_BEST_UARs END\n")
    f.write(str(ALL_BEST_UARS_END) + "\n")
    f.write("ALL_BEST_WARs END\n")
    f.write(str(ALL_BEST_WARS_END) + "\n")
    f.write("ALL_BEST_F1s END\n")
    f.write(str(ALL_BEST_F1_END) + "\n")

    f.write("---\n")
    f.write(f"Average Best UAR mid: {np.mean(ALL_BEST_UARS_MID)}\n")
    f.write(f"Average Best WAR mid: {np.mean(ALL_BEST_WARS_MID)}\n")
    f.write(f"Average Best F1 mid: {np.mean(ALL_BEST_F1_MID)}\n")
    f.write("---\n")
    f.write(f"Average Best UAR end: {np.mean(ALL_BEST_UARS_END)}\n")
    f.write(f"Average Best WAR end: {np.mean(ALL_BEST_WARS_END)}\n")
    f.write(f"Average Best F1 end: {np.mean(ALL_BEST_F1_END)}\n")
