
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from scipy import integrate
import os




class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture with 3 downsampling and 3 upsampling layers."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        # Decoding layers where the resolution increases
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense4 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense5 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Print input shapes
        print("Input x shape:", x.shape)
        print("Input t shape:", t.shape)

        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        print("Embedding shape:", embed.shape)

        # Encoding path
        h1 = self.conv1(x)
        print("After conv1 shape:", h1.shape)
        # Incorporate information from t
        h1 += self.dense1(embed)
        print("After dense1 shape:", h1.shape)
        # Group normalization
        h1 = self.gnorm1(h1)
        print("After gnorm1 shape:", h1.shape)
        h1 = self.act(h1)
        print("After activation1 shape:", h1.shape)

        h2 = self.conv2(h1)
        print("After conv2 shape:", h2.shape)
        h2 += self.dense2(embed)
        print("After dense2 shape:", h2.shape)
        h2 = self.gnorm2(h2)
        print("After gnorm2 shape:", h2.shape)
        h2 = self.act(h2)
        print("After activation2 shape:", h2.shape)

        h3 = self.conv3(h2)
        print("After conv3 shape:", h3.shape)
        h3 += self.dense3(embed)
        print("After dense3 shape:", h3.shape)
        h3 = self.gnorm3(h3)
        print("After gnorm3 shape:", h3.shape)
        h3 = self.act(h3)
        print("After activation3 shape:", h3.shape)

        # Decoding path
        h = self.tconv3(h3)
        print("After tconv3 shape:", h.shape)
        # Skip connection from the encoding path
        h += self.dense4(embed)
        print("After dense4 shape:", h.shape)
        h = self.tgnorm3(h)
        print("After tgnorm3 shape:", h.shape)
        h = self.act(h)
        print("After activation4 shape:", h.shape)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        print("After tconv2 shape:", h.shape)
        h += self.dense5(embed)
        print("After dense5 shape:", h.shape)
        h = self.tgnorm2(h)
        print("After tgnorm2 shape:", h.shape)
        h = self.act(h)
        print("After activation5 shape:", h.shape)

        h = self.tconv1(torch.cat([h, h1], dim=1))
        print("After tconv1 shape:", h.shape)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        print("After normalization shape:", h.shape)

        return h







def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=device)




#@title Define the loss function (double click to expand or collapse)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    #@title Training (double click to expand or collapse)

    # 如果cuda不可用，将device设置为cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sigma = 25.0  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
    score_model = score_model.to(device)

    n_epochs = 50#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size = 64 #@param {'type':'integer'}
    ## learning rate
    lr = 1e-4 #@param {'type':'number'}



    def generate_data(N, T, num_samples):
        # 生成基础信号部分 (利用广播机制)
        k = np.arange(1, N + 1)[:, None]  # 形状 (N, 1)
        t = np.arange(T) * np.pi / 48  # 形状 (T,)

        # 计算正弦信号 (N, T)
        signal = k * np.sin(t)  # 广播乘法

        # 添加通道维度并扩展样本维度 (1, N, T) -> (num_samples, 1, N, T)
        signal = signal.reshape(1, 1, N, T)
        signal = np.repeat(signal, num_samples, axis=0)

        # 生成噪声 (num_samples, 1, N, T)
        noise = np.random.normal(0, 0.1, (num_samples, 1, N, T))

        # 合并信号和噪声
        data = signal + noise

        return torch.tensor(data, dtype=torch.float32)


    # 示例：生成数据
    data = generate_data(N=9, T=96, num_samples=1000)
    print(data.shape)  # 输出应为 [1000, 1, 9, 96]
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = tqdm(range(n_epochs))
    # 保存模型的路径
    models_file_path = 'sde_checkpoints'  # 保存模型的目录
    os.makedirs(models_file_path, exist_ok=True)  # 创建目录（如果不存在）

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x[0].to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        models_path = os.path.join(models_file_path, f'sde_{epoch}.pth')
        torch.save(score_model.state_dict(), models_path)

