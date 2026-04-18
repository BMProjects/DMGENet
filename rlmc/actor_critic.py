import torch
import torch.nn as nn
import torch.nn.functional as F
from models.causal_cnn import CausalCNNEncoder
from collections import deque
import random
import numpy as np
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(
            depth=3,
            kernel_size=2,
            in_channels=obs_dim,
            channels=hidden_dim,
            out_channels=hidden_dim,
            reduced_size=hidden_dim,
        )

        self.rank_embedding = nn.Embedding(action_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, model_loss):
        B, N, T, f = obs.size()
        obs = obs.permute(0, 3, 1, 2).reshape(B, f, N * T)

        ts_emb = self.cnn_encoder(obs)

        # Rank-based error embedding: argsort-of-argsort converts raw errors to
        # integer ranks, which stabilises training when error magnitudes drift.
        rank = torch.argsort(torch.argsort(model_loss, dim=1), dim=1)
        rank_emb = self.rank_embedding(rank).mean(dim=1)

        x = torch.cat([ts_emb, rank_emb], dim=1)
        x = self.net(x)
        return x   # logits [B, action_dim]
    
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(
            depth=3,
            kernel_size=2,
            in_channels=obs_dim,
            channels=hidden_dim,
            out_channels=hidden_dim,
            reduced_size=hidden_dim,
        )

        self.rank_embedding = nn.Embedding(action_dim, hidden_dim)

        self.q_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, model_loss, action):
        B, N, T, f = obs.size()
        obs = obs.permute(0, 3, 1, 2).reshape(B, f, N * T)

        ts_emb = self.cnn_encoder(obs)

        # Must mirror the Actor's rank embedding exactly for Q(s, a) to be consistent.
        rank = torch.argsort(torch.argsort(model_loss, dim=1), dim=1)
        rank_emb = self.rank_embedding(rank).mean(dim=1)

        state_emb = torch.cat([ts_emb, rank_emb], dim=1)

        sa = torch.cat([state_emb, action], dim=1)
        q = self.q_net(sa)
        return q.squeeze(-1) # [B]
    


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, e, a, r, ns, ne, d):
        self.buffer.append((s, e, a, r, ns, ne, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s, e, a, r, ns, ne, d = map(np.array, zip(*batch))

        return (
            torch.FloatTensor(s),
            torch.FloatTensor(e),
            torch.FloatTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(ns),
            torch.FloatTensor(ne),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)
    

class RLMC_env:
    def __init__(self, data_x, data_error, data_y, bm_pred, action_dim):
        self.data_x = data_x
        self.data_error = data_error
        self.data_y = data_y
        self.bm_pred = bm_pred
        self.action_dim = action_dim
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        reward = self._reward(action)

        self.current_step += 1
        done = self.current_step >= len(self.data_x)

        if not done:
            obs, err = self._get_state()
        else:
            # Return zero-padded state rather than None so the ReplayBuffer's
            # np.array(zip(*batch)) call doesn't choke on mixed None/array rows.
            obs = np.zeros_like(self.data_x[0])
            err = np.zeros_like(self.data_error[0])

        return obs, err, reward, done, {}

    def _get_state(self):
        return self.data_x[self.current_step], self.data_error[self.current_step]
    
    def _reward(self, action):
        target = self.data_y[self.current_step]

        pred = np.sum(
            action.reshape(self.action_dim, 1, 1)
            * self.bm_pred[self.current_step],
            axis=0,
        )

        return -np.mean(np.abs(target - pred))


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005):

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(DEVICE)

        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(DEVICE)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

    # ------------------------------------------------
    # action
    # ------------------------------------------------
    def select_action(self, obs, err, noise_std=0.02, eval_mode=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        err = torch.FloatTensor(err).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.actor(obs, err)

            if not eval_mode:
                noise = torch.randn_like(logits) * noise_std
                logits = logits + noise

            action = torch.softmax(logits, dim=1)

        return action.cpu().numpy()[0]
    # ------------------------------------------------
    # update
    # ------------------------------------------------
    def update(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return

        s, e, a, r, ns, ne, d = buffer.sample(batch_size)

        s = s.to(DEVICE)
        e = e.to(DEVICE)
        a = a.to(DEVICE)
        r = r.to(DEVICE)
        ns = ns.to(DEVICE)
        ne = ne.to(DEVICE)
        d = d.to(DEVICE)

        # ---------- target ----------
        with torch.no_grad():
            next_logits = self.target_actor(ns, ne)
            next_action = torch.softmax(next_logits, dim=1)

            target_q = self.target_critic(ns, ne, next_action)
            y = r + self.gamma * (1 - d) * target_q

        # ---------- critic ----------
        q = self.critic(s, e, a)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---------- actor ----------
        logits = self.actor(s, e)
        action = torch.softmax(logits, dim=1)

        actor_loss = -self.critic(s, e, action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------- soft update ----------
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)








