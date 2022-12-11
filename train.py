import argparse
import gym
import torch
import numpy as np
import time
import os
import datetime
import random
from tqdm import tqdm
from utils import Path, plot_results
from ppo import PPOAgent

import pybulletgym

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./runs/Half-Cheetah_" + time.strftime("%Y_%m_%d_%H_%M") + str(random.randint(1, 100)))
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"

def setup(args):
    """
    Setups gym enviroment and agent class
    Input: arguments (dict)
    Ouput: gym enviroment, agent class
    """
    env = gym.make(args['env'])
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    args['input_dim'] = obs_shape
    args['output_dim'] = action_shape
    args['device'] = device
    # breakpoint()
    agent = PPOAgent(env, args, args['input_dim'], args['output_dim'])
    agent.pi.train().to(device)
    agent.critic.train().to(device)
    return env, agent


def train_ppo_agent(args):
    """
    Reinforcement Learning training function
    """
    env, agent = setup(args)
    training_time = time.time()
    cum_rewards = list()
    average_reward_100 = 0
    successfulTrain = False
    eval_rewards = list()
    eval_rewards_mean = 0
    cum_rewards = []
    training_time = time.time()
    average_reward_100 = 0
    successfulTrain = False
    training_rewards, testing_rewards, mean_100_rewards, actor_loss, critic_loss, epochs = [], [], [], [], [], []
    for epoch in range(args['epochs']):
        path = Path()
        done = False
        if args['render']:
            env.render()
        obs = env.reset()
        i = 0
        epoch_return = 0
        # breakpoint()
        while(not done and i < args["steps"]): 
            state = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32, device=device)
            action, action_logprob, _ = agent.forward(state)
            value = agent.critic(state)
            next_obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            epoch_return += reward
            path.add(obs, action.cpu(), next_obs, reward, done, action_logprob.item(), value.item())
            if done == True:
                break
            obs = next_obs
            agent.timesteps += 1
            i += 1

        epochs.append(epoch)
        cum_rewards.append(epoch_return)
        training_rewards.append(epoch_return)
        loss_pi, loss_critic = agent.update(path.obs, path.action, path.reward, path.next_obs, path.done, path.action_logprob, path.value)
        actor_loss.append(loss_pi.detach().numpy())
        critic_loss.append(loss_critic.detach().numpy())
         
        if epoch % args['test_every_epoch'] == 0:
            eval_rewards = eval(env, agent, args)
            writer.add_scalar('test_rewards', eval_rewards, epoch)
            testing_rewards.append(eval_rewards)
        if len(cum_rewards) > 100:
            average_reward_100 = np.array(cum_rewards[-100:]).mean()
            # print(f"{epoch}: average_100 rewards {average_reward_100}")
            writer.add_scalar('mean_100_rewards/train', average_reward_100, epoch)
            mean_100_rewards.append(average_reward_100)

        # print(f"[{epoch}]: loss_pi {loss_pi} \t loss_critic {loss_critic}")
        # print(f"[{epoch}]: epoch_return: {epoch_return}")

        writer.add_scalar('training_rewards', epoch_return, epoch)
        writer.add_scalar('loss_actor/train', loss_pi, epoch)
        writer.add_scalar('loss_critic/train', loss_critic, epoch)

        if average_reward_100 > 2000 or epoch >= args['epochs'] - 1:
            successfulTrain = True
            break
        # if epoch % args['plot_results_itr'] == 0 and epoch != 0:
        #     plot_results(training_rewards, testing_rewards, actor_loss, critic_loss, epochs)

    end_train_time = time.time() - training_time
    if successfulTrain:
            os.makedirs(args['save_dir'], exist_ok=True)
            print("MODEL HAS BEEN SAVE")
            print(f"total train time -> {end_train_time}")
            NOWTIMES = datetime.datetime.now()
            curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
            filenameActor = "half_cheetah_actor_weights_skoutaa_" + str(curr_time) + ".pt"
            filenameCritic = "half_cheetah_critic_weights_skoutaa_" + str(curr_time) + ".pt"
            filenameCombined = "half_cheetah_weights_skoutaa_" + str(curr_time) + ".pt"
            pathActor = os.path.join(args['save_dir'], filenameActor)
            pathCritic = os.path.join(args['save_dir'], filenameCritic)
            pathCombined = os.path.join(args['save_dir'], filenameCombined)
            params = save_params(agent, args)
            torch.save(agent.pi.state_dict(), pathActor)
            torch.save(agent.critic.state_dict(), pathCritic)
            torch.save(params, pathCombined)
            print(f"combined actor-critic model weights path: {pathCombined}")
    env.close()
    writer.close()

    return training_rewards, testing_rewards, mean_100_rewards, actor_loss, critic_loss, epochs


def save_params(agent, args):
    """
    Helper function to combine the weights of the critic and actor networks
    Input: agent class
    Output: network weights (dict)
    """
    params = {
        "actor_state_dict" : agent.pi.state_dict(),
        "critic_state_dict" : agent.critic.state_dict(),
        "args" : args
    }
    return params

def eval(env, agent, args):
    """
    Function that evaulates the agent performance with no noise
    Input: agent class, args (dict)
    Output: testing reward (float)
    """
    result = []
    steps = 0
    dones = 0
    for epoch in range(args['eval_epoch']):
        obs = env.reset()
        done = False
        epoch_return = 0.0
        done = False
        while not done and args['eval_steps'] > steps:
            if args['render']:
                env.render()
            # breakpoint()
            state = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32, device=device)
            action, _, _ = agent.forward(state, noise=False)
            next_obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            epoch_return += reward
            obs = next_obs
            steps += 1
            if done == True:
                dones += 1
        result.append(epoch_return)
        if dones > 1:
            break
    result = np.array(result).reshape(-1,1)
    # print(f"!!!!!!!EVAL Mean ->>{np.mean(result)}")
    return np.mean(result)

def main():
    pass
    # pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ppo")
    # parser.add_argument('-env', type=str, default='HalfCheetahMuJoCoEnv-v0')
    # parser.add_argument('-act', "--activation", type=str, default="ReLU")
    # parser.add_argument('-l', '--layers', type=list, default=[126, 126, 126])
    # parser.add_argument('-ep', '--epochs', type=int, default=20000)
    # parser.add_argument('-st', '--steps', type=int, default=2048)
    # parser.add_argument('-g', '--gamma', type=float, default=0.99)
    # parser.add_argument('-b', '--batch', type=int, default=64)
    # parser.add_argument('-ti', '--test_every_epoch', type=int, default=10)
    # parser.add_argument('-eval_ep', '--eval_epoch', type=int, default=10)
    # parser.add_argument('-eval_st', '--eval_steps', type=int, default=5000)
    # parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    # parser.add_argument("--save_dir", type=str, default='./models/')
    # parser.add_argument("-entr", "--entropy", type=float, default=0.01)
    # parser.add_argument("-k", "--policy_updates", type=int, default=10)
    # parser.add_argument("-a_lr", "--actor_lr", type=float, default=2e-5)
    # parser.add_argument("-c_lr", "--critic_lr", type=float, default=2e-5)
    # parser.add_argument("-clip", "--esp_clip", type=float, default=0.1)
    # parser.add_argument("-grad_c", "--grad_clip", type=float, default=0.5)

    # args = vars(parser.parse_args())
    # print("Hyperparameters -> ", args)
    # train_ppo_agent(args)



if __name__ == "__main__":
    pass
    # main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ppo")
    # parser.add_argument('-env', type=str, default='HalfCheetahMuJoCoEnv-v0')
    # parser.add_argument('-act', "--activation", type=str, default="ReLU")
    # parser.add_argument('-l', '--layers', type=list, default=[126, 126, 126])
    # parser.add_argument('-ep', '--epochs', type=int, default=50000)
    # parser.add_argument('-st', '--steps', type=int, default=2048)
    # parser.add_argument('-g', '--gamma', type=float, default=0.99)
    # parser.add_argument('-b', '--batch', type=int, default=64)
    # parser.add_argument('-tc', '--test_cycles', type=int, default=2000)
    # parser.add_argument('-ti', '--test_every_epoch', type=int, default=10)
    # parser.add_argument('-eval_ep', '--eval_epoch', type=int, default=10)
    # parser.add_argument('-eval_st', '--eval_steps', type=int, default=5000)
    # parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    # parser.add_argument("--save_dir", type=str, default='./models/')
    # parser.add_argument("-entr", "--entropy", type=float, default=0.02)
    # parser.add_argument("-k", "--policy_updates", type=int, default=15)
    # parser.add_argument("-a_lr", "--actor_lr", type=float, default=2e-5)
    # parser.add_argument("-c_lr", "--critic_lr", type=float, default=2e-5)
    # parser.add_argument("-clip", "--esp_clip", type=float, default=0.1)
    # parser.add_argument("-grad_c", "--grad_clip", type=float, default=0.5)

    # args = vars(parser.parse_args())
    # print("Hyperparameters -> ", args)
    # train(args)