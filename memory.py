import torch.nn as nn
import torch
class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, device="cpu"):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)),
                                   requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        assert (node_idxs >= 0).all() and (node_idxs < len(self.memory)).all(), "Invalid node_ids"
        node_idxs = node_idxs.tolist()
        self.memory[node_idxs] = values

    def update_memory(self, updated_memory, updated_lastupdate):
        self.memory = updated_memory.detach()
        self.last_update = updated_lastupdate

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()