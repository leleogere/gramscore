"""
Attempts at making sure RoPE embeddings are working
"""

import matplotlib.pyplot as plt
import torch
from transformers import RoFormerConfig

from models.components.positional_encoding import ContinuousPositionalEncoding1D
from models.components.roformer import *


# First debug the sinusoidal encodings
d_model = 24
shortest_period = 1
longest_period = 10
pos_enc = ContinuousPositionalEncoding1D(
    d_model=d_model,
    shortest_period=shortest_period,
    longest_period=longest_period,
)

t_min = 0
t_max = 20
n_points = 500
positions = torch.linspace(t_min, t_max, steps=n_points)
sinusoidal_positions = pos_enc.forward(positions)
print(f"{positions.shape = }")
print(f"{sinusoidal_positions.shape = }")
print(f"{pos_enc.div_term.shape =}")
print(f"{pos_enc.div_term =}")

assert d_model % 2 == 0, "d_model must be even"
half_dim = d_model // 2

fig, axes = plt.subplots(half_dim, 2, figsize=(8, half_dim * 2))
fig.suptitle(
    f"PE with periods between {shortest_period} and {longest_period} seconds on"
    f"\n{d_model} dimensions (shown between {t_min} and {t_max}s)",
    fontsize=14,
)
for i in range(d_model):
    axes[i % half_dim, i // half_dim].plot(positions, sinusoidal_positions[:, i])
    trig_fn = "sin" if i < half_dim else "cos"
    period = 2 * torch.pi / pos_enc.div_term[i % half_dim].item()  # 2pi / pulsation
    axes[i % half_dim, i // half_dim].set_title(f"Dim {i} ({trig_fn} with period {period:.2f})")

    # Add vertical lines at multiples of the period
    for multiple in range(1, int(t_max / period) + 1):
        axes[i % half_dim, i // half_dim].axvline(x=multiple * period, color='red', alpha=0.5)

plt.tight_layout()
plt.show()


# %% Now try to see in interaction with RoPE
config = RoFormerConfig(
    hidden_size=d_model,
    bias=False,
    num_attention_heads=1,
)
torch.manual_seed(0)
attention_layer = CustomRoFormerSelfAttention(config=config)
# attention_layer.query_key_value.weight.data *= torch.linspace(0, 10, d_model).repeat(3).unsqueeze(1)
# attention_layer.query_key_value.weight.data /= attention_layer.query_key_value.weight.data.norm()
attention_layer.query_key_value.weight.data = 1000000 * torch.ones_like(
    attention_layer.query_key_value.weight.data
)
# attention_layer.query_key_value.bias.data = torch.ones_like(
#     attention_layer.query_key_value.bias.data
# )
attention_layer.eval()

print(attention_layer.query_key_value.bias)

attention_pe = ContinuousPositionalEncoding1D(
    d_model=d_model // config.num_attention_heads,
    shortest_period=0.1,
    longest_period=60_000.0,
)

n_points = 1000
t_min = 0
t_max = 2
x = torch.randn(n_points, d_model)
# p = torch.randn(1, 1, n_points) * 100
p = torch.linspace(t_min, t_max, n_points)
p_sin = attention_pe.forward(p)
print(p.shape, p_sin.shape)
y = attention_layer.forward(x.unsqueeze(0), sinusoidal_pos=attention_pe(torch.tensor([300.0])))[0]
# print(x.norm(), y.norm())

half_dim = attention_pe.d_model // 2
fig, axes = plt.subplots(half_dim, 2, figsize=(8, half_dim * 2))
fig.suptitle(
    f"PE with periods between {shortest_period} and {longest_period} seconds on"
    f"\n{d_model} dimensions (shown between {t_min} and {t_max}s)",
    fontsize=14,
)
for i in range(attention_pe.d_model):
    axes[i % half_dim, i // half_dim].plot(p, p_sin[:, i])
    trig_fn = "sin" if i < half_dim else "cos"
    period = 2 * torch.pi / attention_pe.div_term[i % half_dim].item()  # 2pi / pulsation
    axes[i % half_dim, i // half_dim].set_title(f"Dim {i} ({trig_fn} with period {period:.2f})")

    # Add vertical lines at multiples of the period
    for multiple in range(1, int(t_max / period) + 1):
        axes[i % half_dim, i // half_dim].axvline(x=multiple * period, color='red', alpha=0.5)

plt.tight_layout()
plt.show()

# %% Some plots

m = 1
tmin, tmax = 0, 2
positions = torch.linspace(tmin, tmax, n_points)  # + torch.randn(n_points) * 0.1
sinusoidal_pos = attention_pe.forward(positions).unsqueeze(0).unsqueeze(1).repeat(m, 1, 1, 1)
# sinusoidal_pos = torch.ones_like(sinusoidal_pos)
x = attention_layer.forward(
    # hidden_states=torch.randn(1, n_points, d_model),
    hidden_states=torch.ones(m, n_points, d_model),
    attention_mask=None,
    sinusoidal_pos=sinusoidal_pos,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_sinusoidal_pos=None,
    output_attentions=True,
)
"""
hidden_states.shape
torch.Size([5, 510, 512])
layer.shape
torch.Size([5, 8, 510, 64])
rotate_half_layer.shape
torch.Size([5, 8, 510, 64])
cos_pos.shape
torch.Size([5, 1, 510, 64])
sin_pos.shape
torch.Size([5, 1, 510, 64])
sinusoidal_pos.shape
torch.Size([5, 1, 510, 64])
"""


from utils.debug import show_matrix

show_matrix(x[0][0], extent=(tmin, tmax, 0, d_model), xlabel="Time (s)", ylabel="Dimension")

# plt.plot(x[0][0, :, 0].detach().numpy())
# plt.show()


# %% Plot attention matrix
show_matrix(x[1][0].mean(0), extent=(tmin, tmax, tmin, tmax), xlabel="Time (s)", ylabel="Time (s)")

# Plot average attention score between two tokens as a function of their relative distance
# x[1] shape: (batch_size, num_heads, seq_len, seq_len)
attention_matrix = x[1].mean(dim=(0, 1))  # Average over batch and heads: (seq_len, seq_len)

# Compute average attention for each relative distance
position_diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # (n_points, n_points)
unique_distances = torch.unique(position_diffs[position_diffs >= 0]).sort()[0]
avg_attention_by_distance = []
distance_values = []
for distance in unique_distances:
    # Get all attention scores at this relative distance
    mask = torch.isclose(position_diffs, distance, atol=1e-6)
    scores = attention_matrix[mask]
    if len(scores) > 0:
        avg_attention_by_distance.append(scores.mean().item())
        distance_values.append(distance.item())

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(distance_values, avg_attention_by_distance)
plt.xlabel('Relative Distance Between Tokens')
plt.ylabel('Average Attention Score')
plt.title('Average Attention Score vs Relative Distance')
plt.grid(True)
plt.show()
