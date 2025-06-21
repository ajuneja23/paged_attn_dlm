import torch


class KVPageManager:
    def __init__(
        self, num_layers: int, num_heads: int, qkv_dim: int, page_size: int, device: str
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.page_size = page_size
        self.num_pages = 0
        self.cur_layer = 0  # next layer to be processed, as K,Vs arrive sequentially in layer order
        self.cur_head = 0  # next head to be processed, as K,Vs arrive sequentially in head order
        self.last_page_size = 0
        self.seq_len = 0
        self.key_pages = [
            [[] for _ in range(num_heads)] for _ in range(num_layers)
        ]  # key_pages[layer_idx][head_idx][page_idx]
        self.value_pages = [[[]
                             for _ in range(num_heads)] for _ in range(num_layers)]
        self.device = device

    def prefill_from_tensor(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        if len(key_tensor.shape) != 4 or len(value_tensor.shape) != 4:
            raise ValueError("Key and value tensors must have 4 dimensions")
        if key_tensor.shape != value_tensor.shape:
            raise ValueError("Key and value tensors must have the same shape")
        if key_tensor.shape[0] != self.num_layers:
            raise ValueError("Wrong num_layers value")
        if key_tensor.shape[1] != self.num_heads:
            raise ValueError("Wrong num_heads value")
        if key_tensor.shape[3] != self.qkv_dim:
            raise ("Wrong qkv_dim value")
        num_tokens = key_tensor.shape[2]
        self.last_page_size = num_tokens % self.page_size
        self.num_pages = num_tokens // self.page_size + \
            (self.last_page_size > 0)
        self.key_pages = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)
        ]
        self.value_pages = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)
        ]
        for i in range(self.num_layers):
            for j in range(self.num_heads):
                for k in range(self.num_pages - 1):
                    self.key_pages[i][j].append(
                        key_tensor[
                            i, j, k * self.page_size: (k + 1) * self.page_size, :
                        ]
                    )
                    self.value_pages[i][j].append(
                        value_tensor[
                            i, j, k * self.page_size: (k + 1) * self.page_size, :
                        ]
                    )
                if self.last_page_size > 0:
                    last_page_keys = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    last_page_value = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    last_page_keys[:self.last_page_size,
                                   :] = key_tensor[i, j, : self.last_page_size, :]
                    last_page_value[:self.last_page_size,
                                    :] = value_tensor[i, j, : self.last_page_size, :]
                    self.key_pages[i][j].append(
                        last_page_keys
                    )
                    self.value_pages[i][j].append(
                        last_page_value
                    )
        self.cur_layer = 0

    # overload if easier to give as list of layer tensors
    # key_list[layer_idx] has shape (num_heads,seq_len,qkv_dim)
    def prefill_from_layer_list(self, key_list: list[torch.Tensor], value_list: list[torch.Tensor]):
        if len(key_list) != len(value_list):
            raise ("Key and Value List must be of same length")
        if len(key_list) != self.num_layers:
            raise ("Key and Value List must be of same length as num_layers")
        key_layer_shape = key_list[0].shape
        for i in range(self.num_layers):
            if key_list[i].shape != key_layer_shape:
                raise ("Shape must be consistent across layers")
            if value_list[i].shape != key_layer_shape:
                raise ("Shape Must be consistent across layers")
        if key_layer_shape[0] != self.num_heads:
            raise ("Wrong num_heads value")
        if key_layer_shape[2] != self.qkv_dim:
            raise ("Wrong qkv_dim value")
        # load class variables
        self.seq_len = key_layer_shape[1]
        self.last_page_size = self.seq_len % self.page_size
        self.num_pages = self.seq_len//self.page_size+(self.last_page_size > 0)
        self.key_pages = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)]
        self.value_pages = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)]
        # load pages
        for i in range(self.num_layers):
            self.cur_layer = i
            for j in range(self.num_heads):
                self.cur_head = j
                for k in range(self.num_pages-1):
                    self.key_pages[i][j].append(
                        key_list[i][j, k *
                                    self.page_size:(k+1)*self.page_size, :]
                    )
                    self.value_pages[i][j].append(
                        value_list[i][j, k *
                                      self.page_size:(k+1)*self.page_size, :]
                    )
            if self.last_page_size > 0:
                last_page_keys = torch.zeros(
                    self.page_size, self.qkv_dim).to(self.device)
                last_page_value = torch.zeros(
                    self.page_size, self.qkv_dim).to(self.device)
                last_page_keys[:self.last_page_size,
                               :] = key_list[i][j, :self.last_page_size, :]
                last_page_value[:self.last_page_size,
                                :] = value_list[i][j, :self.last_page_size, :]
        self.cur_head = 0
        self.cur_layer = 0

    # list of list of tensors, key_list[layer_idx][head_idx] has shape (seq_len,qkv_dim)
    def prefill_from_layer_head_list(self, key_list: list[list[torch.Tensor]], value_list: list[list[torch.Tensor]]):
        if len(key_list) != len(value_list):
            raise ("Key and Value List must be of same length")
        if len(key_list) != self.num_layers:
            raise ("Key and Value List must be of same length as num_layers")
        for i in range(self.num_layers):
            if len(key_list[i]) != len(value_list[i]):
                raise ("Key and Value List must be of same length as num_heads")
            if len(key_list[i]) != self.num_heads:
                raise ("Key and Value List must be of same length as num_heads")
            for j in range(self.num_heads):
                if key_list[i][j].shape[1] != self.qkv_dim or value_list[i][j].shape[1] != self.qkv_dim:
                    raise ("Wrong qkv_dim value")
        self.seq_len = key_list[0][0].shape[0]
        self.last_page_size = key_list[0][0].shape[0] % self.page_size
        self.num_pages = key_list[0][0].shape[0]//self.page_size + \
            (self.last_page_size > 0)
        self.key_pages = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)]
        self.value_pages = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)]
        for i in range(self.num_layers):
            self.cur_layer = i
            for j in range(self.num_heads):
                self.cur_head = j
                for k in range(self.num_pages-1):
                    self.key_pages[i][j].append(
                        key_list[i][j][k*self.page_size:(k+1)*self.page_size, :])
                    self.value_pages[i][j].append(
                        value_list[i][j][k*self.page_size:(k+1)*self.page_size, :])
                if self.last_page_size > 0:
                    last_page_keys = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    last_page_values = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    last_page_keys[:self.last_page_size,
                                   :] = key_list[i][j][:self.last_page_size, :]
                    last_page_values[:self.last_page_size,
                                     :] = value_list[i][j][:self.last_page_size, :]
                    self.key_pages[i][j].append(last_page_keys)
                    self.value_pages[i][j].append(last_page_values)
        self.cur_head = 0
        self.cur_layer = 0

    def add_token_for_layer_head(self, layer_idx: int, head_idx: int, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        if layer_idx != self.cur_layer:
            raise ("Layer index must be the same as the current layer")
        if head_idx != self.cur_head:
            raise ("Head index must be the same as the current head")
        if key_tensor.shape != value_tensor.shape:
            raise ("Key and value tensors must have the same shape")
        if key_tensor.shape[0] != self.qkv_dim or value_tensor.shape[0] != self.qkv_dim:
            raise ("Wrong qkv_dim value")
        if self.last_page_size == self.page_size and self.cur_layer == 0 and self.cur_head == 0:
            for i in range(self.num_layers):
                for j in range(self.num_heads):
                    new_key_page = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    new_value_page = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    self.key_pages[i][j].append(new_key_page)
                    self.value_pages[i][j].append(new_value_page)
            self.last_page_size = 0
            self.num_pages += 1
        insertIdx = self.last_page_size if self.cur_layer == 0 and self.cur_head == 0 else self.last_page_size-1
        self.key_pages[layer_idx][head_idx][-1][insertIdx, :] = key_tensor
        self.value_pages[layer_idx][head_idx][-1][insertIdx, :] = value_tensor
        if self.cur_layer == 0 and self.cur_head == 0:
            self.last_page_size += 1  # update last_page_size in the first instance a token is added
        self.cur_head += 1
        if self.cur_head == self.num_heads:
            self.cur_head = 0
            self.cur_layer += 1
        self.seq_len += 1

    # add all heads at once via a tensor, shape (head_idx,qkv_dim)
    def add_token_for_layer(self, layer_idx: int, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        if self.cur_head != 0:
            raise ("All heads must be added at once")
        if layer_idx != self.cur_layer:
            raise ("Not correct layer")
        if len(key_tensor.shape) != 2:
            raise ("Key tensor must have 2 dimensions")
        if key_tensor.shape != value_tensor.shape:
            raise ("Key and value tensors must have the same shape")
        if key_tensor.shape[0] != self.num_heads:
            raise ("Wrong num_heads value")
        if key_tensor.shape[1] != self.qkv_dim:
            raise ("Wrong qkv_dim value")
        if self.last_page_size == self.page_size and self.cur_layer == 0:  # add new page
            for i in range(self.num_layers):
                for j in range(self.num_heads):
                    key_page = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    value_page = torch.zeros(
                        self.page_size, self.qkv_dim).to(self.device)
                    self.key_pages[i][j].append(key_page)
                    self.value_pages[i][j].append(value_page)
            self.last_page_size = 0
            self.num_pages += 1
        insertIdx = self.last_page_size if self.cur_layer == 0 else self.last_page_size-1
        for head_idx in range(self.num_heads):
            self.key_pages[layer_idx][head_idx][-1][insertIdx,
                                                    :] = key_tensor[head_idx, :]
            self.value_pages[layer_idx][head_idx][-1][insertIdx,
                                                      :] = value_tensor[head_idx, :]
        if self.cur_layer == 0:
            self.last_page_size += 1
        self.cur_layer += 1
        self.seq_len += 1

    def get_pages(self, layer_idx: int, head_idx: int):
        if layer_idx >= self.num_layers or head_idx >= self.num_heads or layer_idx < 0 or head_idx < 0:
            raise ("parameters are not in bound")
        return self.key_pages[layer_idx][head_idx], self.value_pages[layer_idx][head_idx]
