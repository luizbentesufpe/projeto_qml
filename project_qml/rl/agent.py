import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


Transition = namedtuple("Transition", "s a r s1 done")

class QNet(nn.Module):
    def __init__(self, L_max, n_actions):
        super().__init__()
        in_dim = 5 * L_max + 2
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),    nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
    

    def forward(self, x):
        return self.net(x)
    
class NStepBuffer:
    def __init__(self, n): self.n, self.buf = n, deque()

    def push(self, *args): self.buf.append(Transition(*args))

    def is_ready(self): return len(self.buf) >= self.n

    def pop_nstep(self, gamma):
        R = 0.0; s1=None; done=False
        for i, tr in enumerate(self.buf):
            R += (gamma ** i) * tr.r
            s1 = tr.s1; done = tr.done
            if done: break
        first = self.buf[0]
        self.buf.popleft()
        return first.s, first.a, R, s1, done
    

class ReplayBuffer:
    def __init__(self, capacity,device: str = "cpu"): 
        self.buf = deque(maxlen=capacity)
        self.DEVICE = device

    def push(self, *args): self.buf.append(Transition(*args))

    def sample(self, bs):
        bs = int(min(int(bs), len(self.buf)))
        idx = np.random.choice(len(self.buf), bs, replace=False)
        batch = [self.buf[i] for i in idx]
        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=self.DEVICE)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64, device=self.DEVICE).unsqueeze(1)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.DEVICE).unsqueeze(1)
        s1= torch.tensor(np.stack([b.s1 for b in batch]),dtype=torch.float32, device=self.DEVICE)
        d = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.DEVICE).unsqueeze(1)
        return s, a, r, s1, d
    
    def __len__(self): return len(self.buf)


class DDQNAgent:
    def __init__(self, cfg, n_actions: int, device: torch.device | str = "cpu"):
        self.DEVICE = torch.device(device)
        self.cfg = cfg
        self.online = QNet(cfg.L_max, n_actions).to(self.DEVICE)
        self.target = QNet(cfg.L_max, n_actions).to(self.DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=1e-3)
        self.replay = ReplayBuffer(cfg.replay_capacity, device=self.DEVICE)
        self.nbuf = NStepBuffer(cfg.n_steps)
        self.step_count = 0
        self.eps = cfg.eps_start
        self.eps_decay = (cfg.eps_start - cfg.eps_end) / cfg.eps_decay_steps
        self.eps_decay_mult = float(getattr(cfg, "eps_decay_mult", 1.0))


    def select_action(self, state_tensor, valid_mask: np.ndarray):
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return 0
        if random.random() < self.eps:
            return int(np.random.choice(valid_idx))
        with torch.no_grad():
            # garante que o estado está no mesmo device da rede
            if isinstance(state_tensor, torch.Tensor):
                state_tensor = state_tensor.to(self.DEVICE)
            q = self.online(state_tensor).detach().cpu().numpy().reshape(-1)
            q[~valid_mask] = -1e9
            return int(np.argmax(q))

    def update(self, gamma):
        if len(self.replay) < self.cfg.batch_size:
            return
        s, a, r, s1, d = self.replay.sample(self.cfg.batch_size)
        with torch.no_grad():
            a_sel = self.online(s1).argmax(dim=1, keepdim=True)
            q_tgt = self.target(s1).gather(1, a_sel)
            y = r + (1 - d) * gamma * q_tgt
        q = self.online(s).gather(1, a)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.step_count += 1
        if self.step_count % self.cfg.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        if self.eps > self.cfg.eps_end:
            dec = float(self.eps_decay) * float(max(self.eps_decay_mult, 1e-6))
            self.eps = max(self.cfg.eps_end, self.eps - dec)

