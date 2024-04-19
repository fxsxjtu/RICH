import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

datasets = ['NYBnbReservation', 'NYBnbPrice', 'CHBnbReservation', 'CHBnbPrice']


def calu_value(prediction, label):
    rmse = np.sqrt(np.mean((prediction - label) ** 2))
    mae = np.mean(np.abs(prediction - label))
    pcc, _ = pearsonr(prediction, label)
    r2 = r2_score(label, prediction)
    return rmse, mae, pcc, r2


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


for dataset in datasets:
    a = np.load("data/" + dataset + '/train.npz')
    b = np.load("data/" + dataset + '/test.npz')
    c = np.load("data/" + dataset + '/val.npz')

    train_x = a['x'].squeeze(-1)
    train_y = a['y'].squeeze(-1)
    val_x = c['x'].squeeze(-1)
    val_y = c['y'].squeeze(-1)
    test_x = b['x'].squeeze(-1)
    test_y = b['y'].squeeze(-1)
    train_y = torch.tensor(train_y).float().cuda()
    train_x = torch.tensor(train_x).float().cuda()
    val_y = torch.tensor(val_y).float().cuda()
    val_x = torch.tensor(val_x).float().cuda()
    test_y = torch.tensor(test_y).float().cuda()
    test_x = torch.tensor(test_x).float().cuda()

    if dataset in ['NYBnbReservation', 'NYBnbPrice']:
        input_dim = 57
        output_dim = 57
    else:
        input_dim = 77
        output_dim = 77
    hidden_dim = 256

    model = SimpleRNN(input_dim, hidden_dim, output_dim).cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_rmse = 10000
    num_epochs = 2000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output.squeeze(), train_y.squeeze())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if epoch % 100 == 0:
                print(f"Epoch {epoch} Loss: {loss}")
            predictions = model(val_x)
            rmse, mae, pcc, r2 = calu_value(predictions.squeeze().cpu().numpy().reshape(-1),
                                            val_y.cpu().squeeze().numpy().reshape(-1))
            if best_rmse > rmse:
                best_rmse = rmse
                predictions_test = model(test_x)
                test_rmse, test_mae, test_pcc, test_r2 = calu_value(predictions_test.cpu().numpy().reshape(-1),
                                                                    test_y.cpu().squeeze().numpy().reshape(-1))
    print(f"==========The performance on {dataset}==========")
    print(test_rmse, test_mae, test_pcc, test_r2)
    print("=" * 50)
    print("\n" * 2)
