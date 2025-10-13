import time
import argparse
import numpy as np
import torch

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent
from utils.ERbuffer import MyMemoryBuffer
from Environment.init_env import init_env
from tensorboardX import SummaryWriter

from utils.validate import (
    EvaluationConfig,
    create_writer as create_validation_writer,
    evaluate_checkpoint,
    save_csv as save_validation_csv,
)

# MEMORY_SIZE = 40000
# MEMORY_WARMUP_SIZE = 2000
# LEARN_FREQ = 50
# BATCH_SIZE = 512
# LEARNING_RATE = 0.001
# GAMMA = 0.5

writer = SummaryWriter('./models/DQNmodels/DDQNmodels3_23/runs/train_process3_21')


# 启用环境进行训练，done=1则结束该次训练，返回奖励值
def run_train_episode(agent, env, rpmemory, MEMORY_WARMUP_SIZE, LEARN_FREQ, BATCH_SIZE):
    total_reward = 0
    train_loss = 1e8
    state, escapeFlag, info = env.reset()
    step = 0
    while True:
        step += 1
        # 智能体抽样动作
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        # print(f"reward:{reward}\n")
        # print(reward)
        rpmemory.add((state, action, reward, next_state, done))

        # 当经验回放数组中的经验数量足够多时（大于给定阈值，手动设定），每50个时间步训练一次
        if (rpmemory.size() > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            experiences = rpmemory.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)
            # 智能体更新价值网络
            train_loss = agent.learn(batch_state, batch_action, batch_reward, batch_next_state, batch_done)

        total_reward += reward
        state = next_state
        if done != -1:
            break
    return total_reward, train_loss


# 验证环境5次，取平均的奖励值
def run_evaluate_episodes(agent, env, eval_episodes=10, render=False):
    eval_reward = []

    for i in range(eval_episodes):
        state, escapeFlag, info = env.reset()
        episode_reward = 0
        t = 0
        while True:
            # 智能体选取动作执行
            action = agent.predict(state)
            state, reward, done, _ = env.step(action)
            t += 1
            episode_reward += reward
            # if episode_reward < -1000 or episode_reward > 1000:
            #     print(episode_reward)
            # render 在此自建平台不支持，可在自己的计算机上开启
            if render:
                env.render()

            if done != -1:
                t = 0
                break
        eval_reward.append(episode_reward)

    return np.mean(eval_reward)


# 验证环境5次，取平均的奖励值
def run_mean_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []

    for i in range(eval_episodes):
        state, escapeFlag, info = env.reset()
        episode_reward = 0
        t = 0
        """打印奖励R 距离状态dist 动作Act的题目"""
        with open("state.txt", "a") as file:
            file.write("new episode:....................................\n")
            file.write("state for t=i \t ClosestDist \t ClosestMissileIndex\n")
        with open("action.txt", "a") as file:
            file.write("new episode:....................................\n")
        with open("reward.txt", "a") as file:
            file.write("new episode:....................................\n")
        while True:
            # 智能体选取动作执行
            action = agent.predict(state)
            state, reward, done, _ = env.step(action)
            dist, index = env.getClosetMissileDist()
            # if t == 0:
            #     index_tmp = index
            #     action_tmp = action
            #     reward_tmp = reward
            # if index_tmp == index and action_tmp == action and abs(reward_tmp-reward) > 20:
            #     print("reward error attention!!!")
            # index_tmp = index
            # action_tmp = action
            # reward_tmp = reward

            # if dist < 2000:
            #     print(dist)

            """打印奖励R 距离状态dist 动作Act"""
            with open("state.txt", "a") as file:
                file.write(f"state for t={t}: \t")
                file.write(str(dist))
                file.write("\t")
                file.write(str(index))
                file.write("\n")
            with open("action.txt", "a") as file:
                file.write(f"action for t={t}: \t")
                file.write(str(action))
                file.write("\n")
            with open("reward.txt", "a") as file:
                file.write(f"reward for t={t}: \t")
                file.write(str(reward))
                file.write("\n")

            t += 1
            episode_reward += reward
            # if episode_reward < -1000 or episode_reward > 1000:
            #     print(episode_reward)
            # render 在此自建平台不支持，可在自己的计算机上开启
            if render:
                env.render()

            if done != -1:
                episode_reward /= t
                t = 0
                break
        eval_reward.append(episode_reward)

    return np.mean(eval_reward)

def main():

    parser = argparse.ArgumentParser(description='612DD')

    parser.add_argument('--memory_size', type=int, default=60000, help='Size of replay memory')
    parser.add_argument('--memory_warmup_size', type=int, default=4000, help='Warmup size of replay memory')
    parser.add_argument('--learn_freq', type=int, default=20, help='Frequency of learning')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.993, help='Discount factor')
    parser.add_argument('--max_episode', type=int, default=1000, help='Maximum number of episodes')
    parser.add_argument('--target_update_interval', type=int, default=120, help='Target network update frequency (in gradient steps)')
    parser.add_argument('--epsilon_start', type=float, default=0.95, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_final', type=float, default=0.1, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon_decay_steps', type=int, default=1500000, help='Number of environment steps to decay epsilon over')
    parser.add_argument('--gradient_clip_norm', type=float, default=10.0, help='Gradient clipping norm (<=0 disables clipping)')
    parser.add_argument(
        '--validation_episodes',
        type=int,
        default=EvaluationConfig.episodes,
        help='Number of validation episodes to run after each checkpoint save',
    )

    args = parser.parse_args()

    MEMORY_SIZE = args.memory_size
    MEMORY_WARMUP_SIZE = args.memory_warmup_size
    LEARN_FREQ = args.learn_freq
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    epsilon_range = max(0.0, args.epsilon_start - args.epsilon_final)
    if args.epsilon_decay_steps > 0:
        epsilon_decrement = epsilon_range / args.epsilon_decay_steps
    else:
        epsilon_decrement = 0.0

    start = time.time()

    num_missiles = 3
    step_num = 3500
    Env, aircraft, missiles = init_env(
        num_missiles=num_missiles,
        StepNum=step_num,
    )

    action_size = Env._get_actSpace()

    state_size = Env._getNewStateSpace()[0]
    # print(state_size)

    # 初始化经验数组
    rpm = MyMemoryBuffer(MEMORY_SIZE)

    # 生成智能体
    model = Double_DQN(state_size=state_size, action_size=action_size)

    agent = MyDQNAgent(
        model,
        action_size,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        e_greed=args.epsilon_start,
        e_greed_min=args.epsilon_final,
        e_greed_decrement=epsilon_decrement,
        update_target_steps=args.target_update_interval,
        gradient_clip_norm=args.gradient_clip_norm,
    )

    max_episode = 2000

    validation_config = EvaluationConfig(
        episodes=args.validation_episodes,
        num_missiles=num_missiles,
        step_num=step_num,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        epsilon_start=0.0,
        epsilon_final=0.0,
        epsilon_decay_steps=1,
        target_update_interval=args.target_update_interval,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    val_writer, val_log_dir = create_validation_writer()
    validation_results = []
    validation_csv_path = None

    train_loss = 0

    # start training
    start_time = time.time()
    print('start training...')
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(50):
            total_reward, train_loss = run_train_episode(agent, Env, rpm, MEMORY_WARMUP_SIZE, LEARN_FREQ, BATCH_SIZE)
            writer.add_scalar('loss', train_loss, episode)
            episode += 1

        # test part
        if episode % 50 == 0:
            eval_reward = run_evaluate_episodes(agent, Env, render=False)
            eval_mean_reward = run_mean_evaluate_episodes(agent, Env, render=False)
            writer.add_scalar('eval reward', eval_reward, episode)
            writer.add_scalar('eval_mean_reward', eval_mean_reward, episode)
            print('episode:{}    e_greed:{}   Test reward:{}    Mean_Reward:{}   Train Loss:{}'.format(episode, agent.e_greed, eval_reward, eval_mean_reward, train_loss))
        if episode % 100 == 0:
            ## 保存模型
            checkpoint_path = './models/DQNmodels/DDQNmodels3_23/DDQN_episode{}.pth'.format(episode)
            torch.save({'model': model.state_dict()}, checkpoint_path)

            success_rate = evaluate_checkpoint(checkpoint_path, validation_config)
            val_writer.add_scalar('intercept_success_rate', success_rate, episode)
            validation_results.append((episode, checkpoint_path, success_rate))
            validation_csv_path = save_validation_csv(val_log_dir, validation_results)
            print(
                'Validation after episode {}: intercept success rate {:.4f}'.format(
                    episode, success_rate
                )
            )
            print('Validation results saved to {}'.format(validation_csv_path))

    print('all used time {:.2}s = {:.2}h'.format(time.time() - start_time, (time.time() - start_time) / 3600))
    # state, reward, escapeFlag, info = Env.step(15)
    # print(state)
    end = time.time()
    total_train_time = end - start
    print("FINAL")
    print(f"total train time for {max_episode} games = {total_train_time} sec")
    val_writer.close()
    # ag = Aircraft(Env.aircraftList, V, Pitch, Heading, dt=0.01, g=9.6     g.num_unique_cards, g.card_dict, cache_limit, epsilon)

    # state_size = g.num_unique_cards + 1  # playable cards + 1 card on top of play deck
    # action_size = g.num_unique_cards  # playable cards

    # init deep q network (it's just a simple feedforward bro)
    # dqn = DQN(state_size, action_size)


main()
