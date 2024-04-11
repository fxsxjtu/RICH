import os
import pickle
import torch
import torch.nn as nn
import time
import random
import numpy as np
import argparse
from model import RICH
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
dataset_config = {
    "ny_bnb": {
        "data_path": "ny_data/final_data",
        "train_start_day": 0,
        "train_end_day": 56,
        "val_start_day": 56,
        "test_start_day": 77,
        "end_day": 98,
        "n_nodes": 57
    },
    "ch_bnb": {
        "data_path": "ch_data/final_data",
        "train_start_day": 0,
        "train_end_day": 56,
        "val_start_day": 56,
        "test_start_day": 77,
        "end_day": 98,
        "n_nodes": 77
    }
}

class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.pro = torch.nn.ReLU()

    def forward(self, predict, truth):
        mask = (truth < 1)
        mask2 = (predict > 0)
        loss = torch.mean(((predict - truth) ** 2) * ~mask + ((mask2 * predict - truth) ** 2) * mask)
        return loss


def calu_value(prediction, label):
    rmse = np.sqrt(np.mean((prediction - label) ** 2))

    mae = np.mean(np.abs(prediction - label))

    pcc, _ = pearsonr(prediction, label)

    r2 = r2_score(label, prediction)

    return rmse, mae, pcc, r2


def infer(model, taxi_start, taxi_end, taxi_timestamp, taxi_backpoints, labels, start_day, eval_start_day, end_day,
          device, data_config, context_type, context_len, context_embeddings=None):
    model.eval()
    predictions = []
    targets = []
    batch_data = []
    attentions = []
    embeddings_list = []
    n_nodes = data_config["n_nodes"]
    if context_embeddings is None:
        context_embeddings = [model.memory_taxi.memory.detach()]
    for now_slice in range(start_day * 24, end_day * 24):
        taxi_batch_head = taxi_backpoints[now_slice]
        taxi_batch_tail = taxi_backpoints[now_slice + 1]
        batch_data.append([taxi_start[taxi_batch_head:taxi_batch_tail], taxi_end[taxi_batch_head:taxi_batch_tail],
                           taxi_timestamp[taxi_batch_head:taxi_batch_tail]])
        label_index = now_slice // 24
        if now_slice % 24 == 23:
            with torch.no_grad():
                label = labels[label_index + 1]

                if label_index >= context_len - 1:
                    context_label = labels[label_index - context_len + 1:label_index + 1]
                else:
                    context_label = labels[:label_index + 1]
                input_context_embeddings = torch.stack(context_embeddings[-context_len:])

                output, embeddings, attention = model(batch_data, now_slice * 3600, context_type, input_context_embeddings,
                                           context_label)
                context_embeddings.append(embeddings)
                embeddings_list.append(embeddings)
                attentions.append(attention)
                if label_index >= eval_start_day:
                    targets.append(label)
                    predictions.append(output)
            batch_data = []
    predictions = torch.stack(predictions).cpu().detach().numpy()
    targets = np.stack(targets)
    rmses = []
    r2s = []
    pccs = []
    maes = []
    for task_i in range(predictions.shape[-1]):
        rmse, r2, mae, pcc = calu_value(predictions[:, :, task_i].reshape(-1), targets[:, :, task_i].reshape(-1))
        rmses.append(rmse)
        r2s.append(r2)
        maes.append(mae)
        pccs.append(pcc)
    return rmses, r2s, maes, pccs, predictions, targets, attentions, embeddings_list

def infer_flow(model, taxi_start, taxi_end, taxi_timestamp, taxi_backpoints, labels, start_day, eval_start_day, end_day,
          device, data_config, context_type, context_len, context_embeddings=None):
    model.eval()
    predictions = []
    targets = []
    batch_data = []
    if context_embeddings is None:
        context_embeddings = [model.memory_taxi.memory.detach()]
    for now_slice in range(start_day * 24, end_day * 24):
        taxi_batch_head = taxi_backpoints[now_slice]
        taxi_batch_tail = taxi_backpoints[now_slice + 1]
        batch_data.append([taxi_start[taxi_batch_head:taxi_batch_tail], taxi_end[taxi_batch_head:taxi_batch_tail],
                           taxi_timestamp[taxi_batch_head:taxi_batch_tail]])
        label_index = now_slice // 24
        if now_slice % 24 == 23:
            with torch.no_grad():
                label = labels[label_index + 1]

                if label_index >= context_len - 1:
                    context_label = labels[label_index - context_len + 1:label_index + 1]
                else:
                    context_label = labels[:label_index + 1]
                input_context_embeddings = torch.stack(context_embeddings[-context_len:])
                output, embeddings, attentions = model(batch_data, now_slice * 3600, context_type, input_context_embeddings,
                                           context_label)
                context_embeddings.append(embeddings)
                if label_index >= eval_start_day:
                    targets.append(label)
                    predictions.append(output)
            batch_data = []
    predictions = torch.stack(predictions).cpu().detach().numpy()
    targets = np.stack(targets)

    rmses = []
    r2s = []
    pccs = []
    maes = []

    rmse, r2, mae, pcc = calu_value(predictions.reshape(-1), targets.reshape(-1))
    rmses.append(rmse)
    r2s.append(r2)
    maes.append(mae)
    pccs.append(pcc)
    return rmses, r2s, maes, pccs, predictions, targets

def get_data(data_config):
    taxi_path = data_config["data_path"]
    taxi_start = np.load(os.path.join(taxi_path, 'start.npy'))
    taxi_end = np.load(os.path.join(taxi_path, 'end.npy'))
    taxi_timestamp = np.load(os.path.join(taxi_path, 'time.npy'))
    taxi_backpoints = np.load(os.path.join(taxi_path, 'backpoints.npy'))
    amount_labels = np.load(os.path.join(taxi_path, 'bnb_label.npy'))[:data_config["end_day"]+1]
    price_labels = np.load(os.path.join(taxi_path, 'bnb_price.npy'))[:data_config["end_day"]+1]
    T, N = amount_labels.shape
    labels = np.concatenate([amount_labels.reshape([T, N, 1]), price_labels.reshape([T, N, 1])], axis=2)
    od_matrix = np.load(os.path.join(taxi_path, "od_matrix.npy"))
    node_features = np.eye(data_config["n_nodes"])
    return taxi_start, taxi_end, taxi_timestamp, taxi_backpoints, labels, node_features, od_matrix


def train_session(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    train_time = str(datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")[:-3])

    num_epoch = args.n_epoch
    learning_rate = args.lr
    weight_decay = args.weight_decay
    cuda_device = torch.device(args.device)
    hidden_dim = args.hidden_dim
    data_name = args.data
    suffix = args.suffix
    save_dir = f'./checkpoint/{data_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_val_model_{learning_rate}_{data_name}_{args.contextlen}_{suffix}_{args.weight_decay}_{args.select_od_nodes}_{args.hidden_dim}.pth')
    config = dataset_config[data_name]

    train_start_day = config["train_start_day"]
    train_end_day = config["train_end_day"]
    val_start_day = config["val_start_day"]
    test_start_day = config["test_start_day"]
    end_day = config["end_day"]

    taxi_start, taxi_end, taxi_timestamp, taxi_backpoints, labels, node_feature, od_matrix = get_data(config)
    flow_labels = np.stack([np.sum(od_matrix, axis=1), np.sum(od_matrix, axis=2)]).transpose([1, 2, 0])  # T N 2

    flow_labels = np.concatenate((flow_labels, od_matrix[:, :, :args.select_od_nodes]), axis=2)
    taxi_timestamp = torch.FloatTensor(taxi_timestamp).to(device=cuda_device)
    context_len = args.contextlen
    context_type = args.context
    assert np.isnan(flow_labels).any() == False, 'label error'
    taxi_end = [int(x) for x in taxi_end]
    taxi_start = [int(x) for x in taxi_start]
    model = RICH(node_feature, config["n_nodes"], hidden_dim, device=cuda_device).to(device=cuda_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
    loss_func = nn.MSELoss().to(device=cuda_device)
    loss_od = OD_loss().to(device=cuda_device)
    best_rmse = 10000
    best_pcc = 0
    best_mae = 0
    best_r2 = 0
    for i in range(num_epoch):
        model.train()
        print('#' * 20)
        start_time = time.time()
        count = 0
        batch_data = []
        context_embeddings = [model.memory_taxi.memory.detach()]
        for now_slice in range(train_start_day * 24, train_end_day * 24):
            taxi_batch_head = taxi_backpoints[now_slice]
            taxi_batch_tail = taxi_backpoints[now_slice + 1]
            with torch.autograd.set_detect_anomaly(True):
                batch_data.append(
                    [taxi_start[taxi_batch_head:taxi_batch_tail], taxi_end[taxi_batch_head:taxi_batch_tail],
                     taxi_timestamp[taxi_batch_head:taxi_batch_tail]])
                if now_slice % 24 == 23:
                    label_index = now_slice // 24

                    label = flow_labels[label_index + 1]

                    if label_index >= context_len - 1:
                        context_label = flow_labels[label_index - context_len + 1:label_index + 1]
                    else:
                        context_label = flow_labels[:label_index + 1]
                    input_context_embeddings = torch.stack(context_embeddings[-context_len:])
                    output, embeddings, _ = model(batch_data, now_slice * 3600, context_type, input_context_embeddings,
                                               context_label)
                    context_embeddings.append(embeddings)
                    loss = loss_func(output.squeeze().float()[:, :2], torch.tensor(label[:, :2]).float().to(cuda_device)) + loss_od(output.squeeze().float()[:, 2:], torch.tensor(label[:, 2:]).float().to(cuda_device))
                    print(f'epoch={i + 1}, days={label_index}, this step loss={loss}')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_data = []
        rmse, mae, pcc, r2, _, _, _, _ = infer(model, taxi_start, taxi_end, taxi_timestamp, taxi_backpoints, flow_labels,
                                         val_start_day, val_start_day, test_start_day, cuda_device, config,
                                         context_type,
                                         context_len, context_embeddings)
        print('epoch=', i + 1, 'validation rmse= ', rmse, 'this step mae= ', mae, 'this step pcc=', pcc)

        if np.mean(rmse) <= best_rmse:
            torch.save(model.state_dict(), save_path)
            best_rmse = np.mean(rmse)
            best_r2 = np.mean(r2)
            best_mae = np.mean(mae)
            best_pcc = np.mean(pcc)
        end_time = time.time()
        print('epoch_time= ', (end_time - start_time) / 60)
    print('#' * 20)
    print('val best rmse is ', best_rmse)
    print('val best r2 is ', best_r2)
    print('val best mae is ', best_mae)
    print('val best pcc is ', best_pcc)

    model.load_state_dict(torch.load(save_path))
    rmse, mae, pcc, r2, predictions, targets, attentions, embeddings_list = infer(model, taxi_start, taxi_end, taxi_timestamp, taxi_backpoints,
                                                     labels, train_start_day, test_start_day, end_day, cuda_device,
                                                     config, context_type, context_len)

    print('test rmse:', rmse)
    print('test r2:', r2)
    print('test mae:', mae)
    print('test pcc:', pcc)
    print(args.weight_decay, args.lr, args.od_num, args.data, args.suffix)
    save_name = f'prompt{context_type}_{data_name}_{suffix}_{train_time}_{context_len}'
    with open(os.path.join(".", f'{save_name}.txt'), 'a+') as file:
        file.write(str(args))
        file.write(f'\nval best rmse is {best_rmse}\n')
        file.write(f'val best r2 is {best_r2}\n')
        file.write(f'val best mae is {best_mae}\n')
        file.write(f'val best pcc is {best_pcc}\n')
        file.write(f'rmse: {rmse}\n')
        file.write(f'r2: {r2}\n')
        file.write(f'mae: {mae}\n')
        file.write(f'pcc: {pcc}\n')
        if args.predict_flow:
            file.write(f'rmse_flow: {np.mean(rmse_flow)}\n')
            file.write(f'r2_flow: {np.mean(r2_flow)}\n')
            file.write(f'mae_flow: {np.mean(mae_flow)}\n')
            file.write(f'pcc_flow: {np.mean(pcc_flow)}\n')
    os.makedirs("./result/prediction", exist_ok=True)
    np.save(os.path.join("./result/prediction", f'{save_name}.npy'), np.stack([targets, predictions]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BootCF training')
    parser.add_argument('--data', type=str, help='Dataset name', default='ch_bnb')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
    parser.add_argument('--best', type=str, default='', help='Path of the best model')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default="cuda:0", help='Idx for the gpu to use: cpu, cuda:0, etc.')
    parser.add_argument('--contextlen', type=int, default=7, help='How much context to use')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimensions of the messages')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--select_od_nodes', type=int, default=77, help='od_num')
    parser.add_argument('--save_prediction', type=bool, default=False, help='choose whether to save prediction')
    args = parser.parse_args()
    train_session(args)
