import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math




def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum





def ssd(x, dt, A, B, C, D=None, chunk_size=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_grups, d_state)
        C: (batch, length, n_grups, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    batch, seq_length, n_heads, d_head = x.shape
    batch, seq_length, n_groups, d_state = B.shape
    
    assert initial_states.shape == (batch, n_heads, d_head, d_state)
    assert x.dtype == A.dtype == B.dtype == C.dtype
    assert x.shape[1] % chunk_size == 0
    assert n_heads % n_groups == 0, "n_heads must be a multiple of n_groups"

    # Impose the tensors to be of shape: b l h d also for B and C
    if  n_heads != n_groups and n_groups > 1:
        heads_per_group = n_heads // n_groups
        B = repeat(B, "b l g d -> b l (g h_group) d", h_group=heads_per_group)
        C = repeat(C, "b l g d -> b l (g h_group) d", h_group=heads_per_group)

    # discretization
    X = x*dt.unsqueeze(-1)
    A = A*dt

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=chunk_size) 
                  for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])

    initial_states= initial_states.unsqueeze(1)
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    
    if D is not None:
        Y = Y + x*D[:, None]                                                        # Add input projection
    
    return Y, final_state





class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-5, norm_before_gate=False, device=None,
                 dtype=None):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

    def forward(self, x, z=None):
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x_norm * self.weight
        if z is not None and self.norm_before_gate:
            x_norm = x_norm * F.silu(z)
        return x_norm





class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state:        int     = 64,
        d_conv:         int     = 4,
        conv_init               = None,
        expand:         int     = 2,
        headdim:        int     = 32,
        ngroups:        int     = 1,
        A_init_range:   tuple   = (1, 16),
        dt_min:         float   = 0.001,
        dt_max:         float   = 0.1,
        dt_init_floor:  float   = 1e-4,
        dt_limit:       tuple   = (0.0,  float("inf")),
        learnable_init_states   = False,
        bias:           bool    = False,
        conv_bias:      bool    = True,
        # Fused kernel and sharding options
        chunk_size:     int     = 16,
        layer_idx               = None,  # Absorb kwarg for general module
        device                  = None,
        dtype                   = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.d_conv     = d_conv
        self.conv_init  = conv_init
        self.expand     = expand
        self.d_inner    = self.expand * self.d_model
        self.headdim    = headdim
        self.ngroups    = ngroups

        assert self.d_inner % self.headdim == 0
        self.nheads                 = self.d_inner // self.headdim
        self.dt_limit               = dt_limit
        self.learnable_init_states  = learnable_init_states
        self.chunk_size             = chunk_size
        self.layer_idx              = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2*self.d_inner + 2*self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, 
                                 **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels  = conv_dim,
            out_channels = conv_dim,
            bias         = conv_bias,
            kernel_size  = d_conv,
            groups       = conv_dim,
            padding      = d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            init_dims = (self.nheads, self.headdim, self.d_state)
            self.init_states = nn.Parameter(torch.zeros(*init_dims, 
                                                        **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)                                                # B L d_in_proj
        A = -torch.exp(self.A_log)                                              # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) \
                         if self.learnable_init_states else None

        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 
                     2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]


        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, 
                                    self.ngroups * self.d_state], dim=-1)

        y, _ = ssd( rearrange(x, "b l (h p) -> b l h p", p=self.headdim), 
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups), 
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups), 
                    self.D,
                    chunk_size=self.chunk_size,
                    initial_states=initial_states)
        
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out