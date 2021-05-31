import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, roc_curve, auc, average_precision_score


class WeightedFocalLoss(torch.nn.Module):

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.alpha = torch.tensor([alpha, 1 - alpha], device=self.device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.to(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        at = at.to(self.device)
        pt = torch.exp(-bce_loss)
        focal_loss = at * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class MyDataset(Dataset):
    def __init__(self, df_x, df_y):
        super(MyDataset, self).__init__()
        self.x = df_x.values
        self.y = df_y.values
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index) -> T_co:
        return torch.tensor(self.x[index], dtype=torch.float, device=self.device), \
               torch.tensor(self.y[index], dtype=torch.float, device=self.device)


class NNBlock(torch.nn.Module):
    def __init__(self, dropout, in_features, out_features, use_dropout=True):
        super(NNBlock, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.bn = torch.nn.BatchNorm1d(num_features=out_features)
        self.dropout = dropout
        self.use_dropout = use_dropout

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out, inplace=False)
        if self.use_dropout:
            out = F.dropout(out, self.dropout)

        return out


class MyModel(torch.nn.Module):
    def __init__(self, dropout, num_layers, in_features, hidden_features):
        super(MyModel, self).__init__()
        self.num_layers = num_layers
        assert self.num_layers >= 5, "Num layer must be larger or equal 5"
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.first_linear = NNBlock(dropout=dropout, in_features=self.in_features,
                                    out_features=int(self.hidden_features / 2), use_dropout=False)
        self.mid_linear_1 = NNBlock(dropout=dropout, in_features=int(self.hidden_features / 2),
                                    out_features=self.hidden_features, use_dropout=True)

        self.mid_linear_2 = NNBlock(dropout=dropout, in_features=self.hidden_features,
                                    out_features=self.hidden_features, use_dropout=True)

        self.mid_linear_3 = NNBlock(dropout=dropout, in_features=self.hidden_features,
                                    out_features=int(self.hidden_features / 2), use_dropout=True)

        self.last_linear = torch.nn.Linear(in_features=int(self.hidden_features / 2), out_features=1)

        self.model = torch.nn.ModuleList()
        self.model.append(self.first_linear)
        self.model.append(self.mid_linear_1)
        for i in range(num_layers - 4):
            self.model.append(self.mid_linear_2)

        self.model.append(self.mid_linear_3)
        self.model.append(self.last_linear)

    def forward(self, x):
        out = self.model[0](x)
        for i in range(1, len(self.model) - 1, 1):
            out = self.model[i](out)

        out = self.model[-1](out)

        return out


class Trainer:
    def __init__(self, x_train_df, y_train_df, x_test_df, y_test_df, n_epochs, in_features=10,
                 hidden_features=64, num_layers=5, batch_size=64, drop_out=0.2, lr=0.01, alpha=0.25):
        self.train_dataset = MyDataset(x_train_df, y_train_df)
        self.test_dataset = MyDataset(x_test_df, y_test_df)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        self.n_epoches = n_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MyModel(dropout=drop_out, num_layers=num_layers, hidden_features=hidden_features,
                             in_features=in_features)
        self.model = self.model.to(self.device)

        self.loss_fn = WeightedFocalLoss(alpha=alpha)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self):
        self.model.train()
        loss_train = 0

        for step, batch in enumerate(self.train_dataloader):
            batch_X, batch_y = batch
            self.optimizer.zero_grad()

            logit = self.model(batch_X)
            logit = torch.squeeze(logit, dim=1)
            loss = self.loss_fn(logit, batch_y)
            loss_train += loss.item()

            loss.backward()
            self.optimizer.step()

        return loss_train

    def train(self, log=False, filelog=None):
        for epoch in range(self.n_epoches):
            loss = self.train_one_epoch()

            train_res = self.evaluate(mod=1)
            test_res = self.evaluate(mod=0)
            print("Epoch {} AUC Training: {:.4f} AP Training {:.4f}".format(epoch+1, train_res[0], train_res[1]))
            print("\t \t AUC Validating: {:.4f} AP Validating {:.4f}".format(test_res[0], test_res[1]))

        print("Training Done")

    def evaluate(self, mod=0):
        self.model.eval()
        loss_train = 0
        y_label = list()
        y_pred = list()

        if mod ==0:
            dataloader = self.test_dataloader
        else:
            dataloader = self.train_dataloader

        for step, batch in enumerate(self.test_dataloader):
            batch_X, batch_y = batch
            y_label.append((batch_y.detach().cpu()))

            with torch.no_grad():
                logit = self.model(batch_X)
                logit = torch.squeeze(logit, dim=1)
                pred = torch.sigmoid(logit)

                y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_label = torch.cat(y_label, dim=0)
        fpr, tpr, thresholds = roc_curve(y_label.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        area_under_curve = auc(fpr, tpr)
        ap = average_precision_score(y_label.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

        return area_under_curve, ap


from preprocess_data import split_dataset

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = split_dataset('healthcare-dataset-stroke-data.csv')
    trainer = Trainer(x_train, y_train, x_test, y_test, n_epochs=10, alpha=25)
    trainer.train()
    result = trainer.evaluate()
    print(result)
