import torch
import torch.nn as nn
import einops


class _LSTM(nn.LSTM):
    """
    Tweaks to PyTorch LSTM module
    * Up the remembering
    """

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # https://danijar.com/tips-for-training-recurrent-neural-networks/
        # forget += 1
        # ifgo
        value = 2#2.0
        idx_input = slice(0, self.hidden_size)
        idx_forget = slice(self.hidden_size, 2 * self.hidden_size)
        for layer in range(self.num_layers):
            for input in ("i", "h"):
                # Balance out the scale of the cell w/ a -=1
                getattr(self, f"bias_{input}h_l{layer}").data[idx_input] -= value
                getattr(self, f"bias_{input}h_l{layer}").data[idx_forget] += value


class SimpleAmpLSTM(nn.Module):               

    def __init__(self, hidden_size=32, burn_in_offset=256, train_truncate=1024):
        super().__init__()
        self.burn_in_offset = burn_in_offset
        self._train_truncate = train_truncate
        self.h_0 = nn.Parameter( torch.zeros((1, hidden_size)) )
        self.c_0 = nn.Parameter( torch.zeros((1, hidden_size)) )

        self.lstm = _LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        
        # Proiezione finale: da Hidden State (32) a Audio Sample (1)
        self.head = nn.Linear(hidden_size, 1)


        self._init_state_burn_in = 48_000


    def forward_train(self, x, hidden_states):
        output_features_list = []
        if self.burn_in_offset > 0:                                                 # Don't backprop through the burn-in
            last_output_features, hidden_states = self.lstm(
                x[:, :self.burn_in_offset, :], hidden_states
            )
            output_features_list.append(last_output_features.detach())
        
        for i in range(self.burn_in_offset, x.shape[1], self._train_truncate):
            if i != self.burn_in_offset:                                            # we detach only after the first iteration
                hidden_states = tuple(h.detach() for h in hidden_states)
            last_output_features, hidden_states = self.lstm(
                x[:, i : i + self._train_truncate, :], hidden_states
            )
            output_features_list.append(last_output_features)
            
        output_features = torch.cat(output_features_list, dim=1)
        out = self.head(output_features)
        return out
    

    def forward_eval(self, x, hidden_states):
        # PyTorch MPS bug: LSTMs silently fail on sequences > 65535
        # We must process the audio in blocks to be safe.
        BLOCK_SIZE = 65535 //2
        output_features_list = []
        
        # Loop through the whole file in chunks of 65535
        for i in range(0, x.shape[1], BLOCK_SIZE):
            out, hidden_states = self.lstm(
                x[:, i : i + BLOCK_SIZE, :], hidden_states
            )
            output_features_list.append(out)
            
        output_features = torch.cat(output_features_list, dim=1)
        out = self.head(output_features)

        return out
    

    def forward(self, x: torch.tensor):
        if x.ndim == 2:
            x = x[:, :, None]

        B, T, one = x.shape

        h_0 = einops.repeat(self.h_0, "1 H -> 1 B H", B=B).contiguous()
        c_0 = einops.repeat(self.c_0, "1 H -> 1 B H", B=B).contiguous()
        hidden_states = (h_0, c_0)

        if self.training:
            # return self.forward_eval(x, hidden_states)
            return self.forward_train(x, hidden_states)
        else:
            # print('eval')
            hidden_states = self._get_initial_state(hidden_states, x.device, B)
            return self.forward_eval(x, hidden_states)
        

    def _get_initial_state(self, hidden_states, device, B, inputs=None):
        if inputs is None:
            inputs = torch.zeros((B, self._init_state_burn_in, 1), device=device)
        
        _, (h, c) = self.lstm(inputs, hidden_states)
        return h, c