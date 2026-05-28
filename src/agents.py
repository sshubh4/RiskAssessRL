"""All agent definitions: Random, DQN, DDQN, A2C, PPO."""
from __future__ import annotations
import math, random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Shared network building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        probs = F.softmax(self.actor(h), dim=-1)
        value = self.critic(h)
        return probs, value


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        return self.env.action_space.sample()


# ---------------------------------------------------------------------------
# Replay memory (shared by DQN / DDQN)
# ---------------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, k: int):
        return random.sample(self.memory, k)

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64,
                 lr: float = 2.5e-4, gamma: float = 0.99,
                 memory_size: int = 10_000, batch_size: int = 128,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.001,
                 epsilon_decay: int = 50_000, target_update: int = 10,
                 optimize_every: int = 5):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.optimize_every = optimize_every
        self.step_count = 0
        self.ep_count = 0

        self.policy_net = MLP(obs_dim, n_actions, hidden).to(device)
        self.target_net = MLP(obs_dim, n_actions, hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        self.criterion = nn.MSELoss()

    @property
    def epsilon(self) -> float:
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
               math.exp(-self.step_count / self.epsilon_decay)

    def act(self, obs: np.ndarray) -> int:
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.policy_net(s).argmax(1).item()

    def remember(self, state, action, next_state, reward):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor([[action]], dtype=torch.long)
        r = torch.tensor([reward], dtype=torch.float32)
        ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) \
             if next_state is not None else None
        self.memory.push(s, a, ns, r)

    def optimize(self):
        if self.step_count % self.optimize_every != 0:
            return
        if len(self.memory) < self.batch_size:
            return

        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        non_final = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        sb = torch.cat(batch.state).to(device)
        ab = torch.cat(batch.action).to(device)
        rb = torch.cat(batch.reward).to(device)

        q = self.policy_net(sb).gather(1, ab)
        next_q = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            if non_final.shape[0] > 0:
                next_q[mask] = self._next_q(non_final, mask)
        target = (next_q * self.gamma) + rb

        loss = self.criterion(q, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _next_q(self, non_final: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.target_net(non_final).max(1).values

    def episode_end(self):
        self.ep_count += 1
        if self.ep_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.policy_net.eval()


# ---------------------------------------------------------------------------
# Double DQN (only _next_q changes)
# ---------------------------------------------------------------------------

class DoubleDQNAgent(DQNAgent):
    def _next_q(self, non_final: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        best_actions = self.policy_net(non_final).argmax(1, keepdim=True)
        return self.target_net(non_final).gather(1, best_actions).squeeze(1)


# ---------------------------------------------------------------------------
# A2C
# ---------------------------------------------------------------------------

class A2CAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256,
                 lr: float = 1e-4, gamma: float = 0.99):
        self.gamma = gamma
        self.model = ActorCritic(obs_dim, n_actions, hidden).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, value = self.model(s)
        dist = torch.distributions.Categorical(probs.squeeze(0))
        action = dist.sample().item()
        return action, value.item()

    def act_greedy(self, obs: np.ndarray) -> int:
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = self.model(s)
        return probs.argmax(1).item()

    def update(self, trajectory: list[tuple]):
        states, actions, rewards, values = zip(*trajectory)
        R, returns = 0.0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.tensor(values, dtype=torch.float32).to(device)
        advantages = returns_t - values_t

        policy_loss, value_loss, entropy_loss = [], [], []
        for s, a, adv, ret in zip(states, actions, advantages, returns_t):
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            probs, val = self.model(st)
            logp = torch.log(probs.squeeze()[a] + 1e-8)
            ent = -(probs * torch.log(probs + 1e-8)).sum()
            policy_loss.append(-logp * adv.detach())
            value_loss.append((ret - val.squeeze()) ** 2)
            entropy_loss.append(-0.01 * ent)

        loss = (torch.stack(policy_loss).sum()
                + torch.stack(value_loss).sum()
                + torch.stack(entropy_loss).sum())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256,
                 lr: float = 1e-4, gamma: float = 0.99,
                 clip: float = 0.2, ppo_epochs: int = 4):
        self.gamma = gamma
        self.clip = clip
        self.ppo_epochs = ppo_epochs
        self.model = ActorCritic(obs_dim, n_actions, hidden).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, obs: np.ndarray):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, value = self.model(s)
        dist = torch.distributions.Categorical(probs.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze(0).item()

    def act_greedy(self, obs: np.ndarray) -> int:
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = self.model(s)
        return probs.argmax(1).item()

    def update(self, trajectory: list[tuple]):
        states, actions, rewards, values, log_probs_old = zip(*trajectory)
        R, returns = 0.0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.stack([v if isinstance(v, torch.Tensor)
                                 else torch.tensor(v) for v in values]).to(device)
        advantages = (returns_t - values_t).detach()
        log_probs_old_t = torch.tensor(log_probs_old, dtype=torch.float32).to(device)

        for _ in range(self.ppo_epochs):
            p_losses, v_losses, e_losses = [], [], []
            for s, a, old_lp, adv, ret in zip(states, actions, log_probs_old_t, advantages, returns_t):
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                probs, val = self.model(st)
                dist = torch.distributions.Categorical(probs.squeeze(0))
                logp = dist.log_prob(torch.tensor(a).to(device))
                ratio = torch.exp(logp - old_lp)
                p_losses.append(-torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv))
                v_losses.append(F.mse_loss(val.squeeze(), ret))
                e_losses.append(-0.01 * dist.entropy())

            loss = torch.stack(p_losses).sum() + torch.stack(v_losses).sum() + torch.stack(e_losses).sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()
