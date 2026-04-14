import torch
import torch.nn as nn
import numpy as np

TARGET_ACTION_DIST = torch.tensor([
    0.08,
    0.10,
    0.10,
    0.10,
    0.12,
    0.10,
    0.08,
    0.14,
    0.14,
    0.07,
    0.07,
], dtype=torch.float32)


def compute_gae(rewards: list[float], values: list[float], dones: list[bool], gamma: float = 0.995, lam: float = 0.95) -> tuple[list[float], list[float]]:
    """
    Computes Generalized Advantage Estimation (GAE) over a collected trajectory

    Args:
        rewards (list[float]): rewards collected at each timestep
        values  (list[float]): critic value estimates at each timestep
        dones   (list[bool]):  episode termination flags at each timestep
        gamma   (float):       discount factor for future rewards
        lam     (float):       GAE lambda, controls bias-variance tradeoff

    Returns:
        tuple containing:
            - advantages (list[float]): GAE advantage estimates per timestep
            - returns    (list[float]): discounted returns (advantage + value) per timestep
    """
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        nxt   = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def _run_ppo_update(model: nn.Module, optimizer: torch.optim.Optimizer, memory, device: torch.device,
                    gamma: float, lam: float, epsilon: float, epochs: int,
                    entropy_coef: float, diversity_coef: float = 0.0) -> tuple[float, float]:
    """
    Runs a full PPO update over the collected memory buffer

    Args:
        model        (nn.Module):   actor-critic network to update
        optimizer    (Optimizer):   torch optimizer tied to the model
        memory       (PPOMemory):   buffer holding the collected trajectory data
        device       (torch.device): CPU or CUDA device to run tensors on
        gamma        (float):       discount factor passed to GAE
        lam          (float):       GAE lambda passed to GAE
        epsilon      (float):       PPO clip range for the probability ratio
        epochs       (int):         number of passes over the collected batch
        entropy_coef (float):       coefficient scaling the entropy bonus in the loss
        diversity_coef (float):     unused coefficient reserved for action diversity loss

    Returns:
        tuple containing:
            - loss_val (float): total loss from the final epoch
            - ent_val  (float): mean policy entropy from the final epoch
    """
    batches = memory.get_batches()

    if memory.use_cnn:
        states_b, cd_b, actions_b, rewards_b, values_b, old_lp_b, dones_b = batches
    else:
        cd_b, actions_b, rewards_b, values_b, old_lp_b, dones_b = batches

    advantages, returns = compute_gae(rewards_b, values_b, dones_b, gamma, lam)
    advantages = np.array(advantages)
    returns = np.array(returns)

    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    loss_val = 0.0
    ent_val = 0.0

    for _ in range(epochs):
        if memory.use_cnn:
            s_t = torch.FloatTensor(states_b).to(device)
            cd_t = torch.FloatTensor(cd_b).to(device)
            probs, vals = model(s_t, cd_t)
        else:
            cd_t = torch.FloatTensor(cd_b).to(device)
            probs, vals = model(cd_t)

        a_t = torch.LongTensor(actions_b).to(device)
        olp_t = torch.FloatTensor(old_lp_b).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)

        probs = torch.clamp(probs, min=1e-6, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist_obj = torch.distributions.Categorical(probs)
        nlp = dist_obj.log_prob(a_t)
        ent = dist_obj.entropy().mean()

        ratio = torch.exp(nlp - olp_t)
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv_t

        actor_loss = -torch.min(s1, s2).mean()
        critic_loss = nn.MSELoss()(vals.squeeze(), ret_t)

        loss = actor_loss + 1.0 * critic_loss - entropy_coef * ent

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_val = loss.item()
        ent_val = ent.item()

    return loss_val, ent_val