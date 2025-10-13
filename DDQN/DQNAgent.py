import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_


# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 更新目标网络的操作函数，在MyDQNAgent.learn()函数中调用
def soft_update(target, source, tau=0):
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # print(target.parameters())
    target.load_state_dict(source.state_dict())
    # print(target.parameters())
        # target_param.set_para(target_param*tau+param*(1.0-tau))


class MyDQNAgent:

    def __init__(
        self,
        model,
        action_size,
        gamma=None,
        lr=None,
        e_greed=0.1,
        e_greed_min=0.1,
        e_greed_decrement=0,
        update_target_steps=80,
        gradient_clip_norm=10.0,
    ):

        self.action_size = action_size
        self.global_step = 0
        self.update_target_steps = max(1, update_target_steps)
        self.e_greed = e_greed  # ϵ-greedy中的ϵ
        self.e_greed_min = max(0.0, e_greed_min)
        self.e_greed_decrement = max(0.0, e_greed_decrement)  # ϵ的动态更新因子
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.gamma = gamma  # 回报折扣因子
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(lr=lr, params=self.model.parameters())
        self.gradient_clip_norm = gradient_clip_norm

    # 使用行为策略生成动作
    def sample(self, state):

        sample = np.random.random()  # [0.0, 1.0)
        if sample < self.e_greed:
            act = np.random.randint(self.action_size)  # 返回[0, action_size)的整数，这里就是0或1
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.action_size)
            else:
                act = self.predict(state)

        # 动态更改e_greed,但不小于0.1
        self.e_greed = max(self.e_greed_min, self.e_greed - self.e_greed_decrement)

        return act

    # DQN网络做预测
    def predict(self, state):

        state = torch.tensor(state, dtype=torch.float32).to(device)
        # DQN网络做预测


        # '''此处将state置为全1数组看是否有影响'''
        # state = torch.ones(state.shape[0],  dtype=torch.float32)
        pred_q = self.model(state).to(device)
        # 选取概率值最大的动作
        act = int(pred_q.argmax())

        # 在概率值前三大的三个动作中选
        # k = 5  # 前k个
        # b = pred_q.detach().numpy().argsort()[-3:][::-1]
        # act = np.random.choice(b, size=1)
        # act = act[0]


        # print(act)
        # act = np.random.randint(act, act + 3)
        return act

    # 更新DQN网络
    def learn(self, state, action, reward, next_state, terminal):
        """Update model with an episode data

        Args:
            state(np.float32): shape of (batch_size, state_size)
            act(np.int32): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
            next_state(np.float32): shape of (batch_size, state_size)
            terminal(np.float32): shape of (batch_size)

        Returns:
            loss(float)
        """

        if self.global_step % self.update_target_steps == 0:
            # 6. 更新目标网络
            self.target_model.load_state_dict(self.model.state_dict())
            # soft_update(self.target_model, self.model)
        self.global_step += 1

        action = np.expand_dims(action, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        state = np.array(state)
        next_state = np.array(next_state)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        # '''此处将state置为全1数组看是否有影响'''
        # state = torch.ones(state.shape, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        terminal = torch.tensor(terminal, dtype=torch.float32).to(device)
        # 1. DQN网络做正向传播
        pred_values = self.model(state)

        # action的维度:2
        action_dim = pred_values.shape[-1]

        # 删除输入action的Shape中尺寸为1的维度
        action = torch.squeeze(action, dim=-1)

        # action进行onhot编码
        action_onehot = nn.functional.one_hot(action.to(torch.int64), num_classes=action_dim)

        pred_value = torch.multiply(pred_values, action_onehot)
        pred_value = torch.sum(pred_value, dim=1, keepdim=True)

        done_mask = (terminal != -1).float()

        # target Q
        with torch.no_grad():
            # 2. 目标网络做正向传播
            next_q_online = self.model(next_state)
            best_next_action = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_model(next_state)
            max_v = next_q_target.gather(1, best_next_action)
            # 3. TD 目标
            target = reward + (1 - done_mask) * self.gamma * max_v
            # print("训练目标")
            # print(target)

        # 4. TD 误差
        loss = self.mse_loss(pred_value, target)

        # 5. 更新DQN的参数
        # 梯度清零
        self.optimizer.zero_grad()
        # 反向计算梯度
        loss.backward()
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        # 梯度更新
        self.optimizer.step()

        return loss
