import torch
import torch.nn as nn
import numpy as np
from memory import Memory



class STBNB(nn.Module):
    def __init__(self, node_feature, n_nodes, emb_size, device):
        super(STBNB, self).__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.emb_size = emb_size
        self.node_feature = torch.from_numpy(node_feature).to(device)
        self.raw_message_dim = (self.node_feature.shape[1] + emb_size) * 2
        self.memory_taxi = Memory(n_nodes=n_nodes, memory_dimension=emb_size, device=device)

        self.hour2day_timemixer = nn.Sequential(nn.Linear(24, 24),
                                                nn.GELU(),
                                                nn.Linear(24, 24))
        self.hour2day_channelinput = nn.Linear(self.raw_message_dim, emb_size)
        self.ln1 = nn.LayerNorm(emb_size, elementwise_affine=False)
        self.hour2day_channelmixer = nn.Sequential(nn.Linear(emb_size, emb_size),
                                                   nn.GELU(),
                                                   nn.Linear(emb_size, emb_size))
        self.updater = nn.GRUCell(emb_size, emb_size)
        self.output_module = nn.Sequential(nn.Linear(emb_size, emb_size // 2),
                                           nn.ReLU(),
                                           nn.Linear(emb_size // 2, emb_size // 2),
                                           nn.ReLU(),
                                           nn.Linear(emb_size // 2, 1))
        self.static_embedding = nn.Embedding(self.n_nodes, emb_size)

        self.wq = nn.Linear(emb_size, emb_size)
        self.wk = nn.Linear(emb_size, emb_size)
        self.wv = nn.Linear(1, emb_size)
        self.wk2 = nn.Linear(1, emb_size)
        self.mlp = nn.Linear(self.n_nodes + emb_size, emb_size)
    def reset_memory(self):
        self.memory_taxi.__init_memory__()


    def get_raw_hour_message(self, taxi_start, taxi_end, taxi_timestamps, now_time, hour_memory=None):
        memory = self.memory_taxi.get_memory()
        taxi_appear = np.concatenate([taxi_start, taxi_end]).reshape([2, -1]).transpose(1, 0)
        appear_unique_ids = np.unique(taxi_appear)
        appear_raw_messages = torch.zeros(len(appear_unique_ids), self.raw_message_dim).to(self.device)
        start_raw_messages = torch.cat([memory[taxi_end], self.node_feature[taxi_end]], dim=1)
        end_raw_messages = torch.cat([memory[taxi_start], self.node_feature[taxi_start]], dim=1)
        time_factor = torch.exp((taxi_timestamps - now_time)/3600).reshape([-1, 1]) # (N,1)
        for i, id_i in enumerate(appear_unique_ids):
            if id_i in taxi_start:
                ind = (taxi_start == id_i)
                ind_start_raw_messages = start_raw_messages[ind]
                aggregated_raw_messages = torch.sum(ind_start_raw_messages * time_factor[ind], dim=0)/torch.sum(time_factor[ind])
                appear_raw_messages[i, :self.raw_message_dim // 2] = aggregated_raw_messages
            if id_i in taxi_end:
                ind = (taxi_end == id_i)
                ind_end_raw_messages = end_raw_messages[ind]
                aggregated_raw_messages = torch.sum(ind_end_raw_messages * time_factor[ind], dim=0)/torch.sum(time_factor[ind])
                appear_raw_messages[i, self.raw_message_dim // 2:] = aggregated_raw_messages
        return appear_unique_ids, appear_raw_messages

    def hour2day(self, tensored_raw_message):
        input_message = self.hour2day_channelinput(tensored_raw_message)  # N T D
        residual = input_message
        across_time = self.hour2day_timemixer(self.ln1(input_message).permute([0, 2, 1])).permute([0, 2, 1]) + residual
        residual2 = across_time
        across_channel = self.ln1(self.hour2day_channelmixer(self.ln1(across_time)) + residual2)
        result = torch.mean(across_channel, dim=1)
        return result, across_channel


    def update_memory(self, yesterday_memory, yesterday_updatetime, day_message, day_unique_ids,
                      day_unique_timestamps):
        unique_updated_memory = self.updater(day_message, yesterday_memory[day_unique_ids])
        yesterday_memory[day_unique_ids] = unique_updated_memory
        yesterday_updatetime[day_unique_ids] = day_unique_timestamps
        return yesterday_memory, yesterday_updatetime

    def day_unique_info(self, batch_data):
        taxi_starts = np.concatenate([batch_i[0] for batch_i in batch_data])
        taxi_ends = np.concatenate([batch_i[1] for batch_i in batch_data])
        taxi_appear = np.concatenate([taxi_starts, taxi_ends]).reshape([2, -1]).transpose(1, 0).reshape([-1])
        taxi_time = torch.cat([batch_i[2] for batch_i in batch_data])
        taxi_time2 = torch.cat([taxi_time, taxi_time]).reshape([2, -1]).transpose(1, 0).reshape([-1])
        day_unique_id = np.unique(taxi_appear)
        last_time = torch.zeros(self.n_nodes).to(self.device)
        last_time[taxi_appear] = taxi_time2
        return day_unique_id, last_time[day_unique_id]

    def node_embedding(self, updated_memory, day_24_messages, updated_lastupdate, now_time, day_unique_ids):
        return updated_memory

    def forward(self, batch_data, now_time, context_type="none", context_embeddings=None, context_labels=None):
        if context_type == "none":
            results = self.output_module(self.static_embedding.weight)
            return results, None
        slices = len(batch_data)
        batch_time = (np.arange(slices) - slices + 1) * 3600 + now_time
        unique_ids = []
        unique_raw_messages = []
        tensored_raw_message = torch.zeros(self.n_nodes, slices, self.raw_message_dim).to(self.device)
        tensored_message_mask = torch.zeros(self.n_nodes, slices).to(self.device)
        for i, (taxi_start, taxi_end, taxi_timestamps) in enumerate(batch_data):
            unique_id, unique_raw_message = self.get_raw_hour_message(taxi_start, taxi_end, taxi_timestamps,
                                                                      batch_time[i])
            unique_ids.append(unique_id)
            unique_raw_messages.append(unique_raw_message)
            tensored_raw_message[unique_id, i] = unique_raw_message
            tensored_message_mask[unique_id, i] = 1
        day_unique_ids, day_unique_timestamps = self.day_unique_info(batch_data)
        day_message, day_24_messages = self.hour2day(tensored_raw_message[day_unique_ids])
        yesterday_memory = self.memory_taxi.memory
        yesterday_updatetime = self.memory_taxi.last_update
        updated_memory, updated_lastupdate = self.update_memory(yesterday_memory, yesterday_updatetime, day_message,
                                                                day_unique_ids, day_unique_timestamps)
        self.memory_taxi.update_memory(updated_memory, updated_lastupdate)
        node_embeddings = self.node_embedding(updated_memory, day_24_messages, updated_lastupdate, now_time, day_unique_ids)
        if context_type == "temporal":
            n_len, n_nodes, n_task = context_labels.shape  # T N 2
            result_list = []
            attention_list = []
            for task_i in range(n_task):
                context_label = context_labels[:, :, task_i]
                label_mean = np.mean(context_label)
                label_std = np.std(context_label)
                context_label = (context_label - label_mean) / label_std
                context_label = torch.FloatTensor(context_label).to(self.device).reshape([n_len * n_nodes, 1])
                Q = self.wq(node_embeddings)  # N d
                K = (self.wk(context_embeddings).reshape([n_len * n_nodes, self.emb_size]) + self.wk2(
                    context_label)).reshape([n_len, n_nodes, self.emb_size])
                V = self.wv(context_label).reshape([n_len, n_nodes, self.emb_size])
                attention_score = torch.einsum("nd,tnd->tn", Q, K) / np.sqrt(self.emb_size)
                attentions = torch.nn.functional.softmax(attention_score, dim=0)
                output = torch.einsum("tn,tnd->nd", attentions, V)
                results = self.output_module(output)
                result_list.append(results * label_std + label_mean)
                attention_list.append(attentions)
            concat_result = torch.cat(result_list, dim=-1) # N 2
            attentions = torch.cat(attention_list)
            return concat_result, node_embeddings.detach().clone(), attentions
