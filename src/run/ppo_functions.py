import torch
import torch.nn as nn
import numpy as np

TARGET_ACTION_DIST = torch.tensor([
    0.08,   # neutral
    0.10,   # move_left
    0.10,   # move_right
    0.10,   # jump
    0.12,   # light
    0.10,   # heavy
    0.08,   # dodge
    0.14,   # left_heavy
    0.14,   # right_heavy
    0.07,   # left_light
    0.07,   # right_light
], dtype=torch.float32)


def compute_gae(rewards, values, dones, gamma=0.995, lam=0.95):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        nxt   = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def _run_ppo_update(model, optimizer, memory, device,
                    gamma, lam, epsilon, epochs,
                    entropy_coef, diversity_coef=0.0):
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