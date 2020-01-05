'''
training PPO algorithm for CarRacing-v0 gym environment
'''

import argparse
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

device = torch.device("cpu")

transitions = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('action_logp', np.float64),
                       ('r', np.float64), ('s_next', np.float64, (args.img_stack, 96, 96))])


class New_Env():
    """
    Modified Environment  
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.num_counter = 0
        self.ave_rwd = rwd_memory()
        
        self.die = False
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack  # state is adjacent 4 frames to get velocity information
        return np.array(self.stack)

    def step_single_image(self, action):
        total_rwd = 0
        for i in range(args.action_repeat): # args.action_repeat default: 8, every action will be repeated for 8 frames
            img_rgb, rwd, die, _ = self.env.step(action)
            if die:
                rwd = rwd + 100  
            
            # penalty grassland area (green)
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                rwd = rwd - 0.05
            total_rwd = total_rwd + rwd
            
            # end episode if no reward received
            if self.ave_rwd(rwd) <= -0.1: 
                done = True
            else:
                done = False

            if done or die:
                break
            
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_rwd, done, die

    def render(self, *arg):
        self.env.render(*arg)


class PPO_Net(nn.Module):
    """
    Policy Gradient Actor-Critic for PPO
    """

    def __init__(self):
        super(PPO_Net, self).__init__()
        self.CNN = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),  
            nn.ReLU(),  
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  
        )  # output shape (256, 1, 1)    
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1)) 
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU()) 
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus()) 
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus()) 
        self.apply(_weights_init)

    def forward(self, x):
        x = self.CNN(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1  
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Car_Agent():     
    def __init__(self):
        self.num_counter = 0
        self.buffer = np.empty(2000, dtype=transitions)

    def store_trasition(self, transitions):
        self.buffer[self.num_counter] = transitions
        self.num_counter += 1
        if self.num_counter == 2000: #buffer_capacity = 2000
            self.num_counter = 0
            return True
        else:
            return False
    def getBuffer(self):
        return self.buffer


def select_action(net, state):
    state = torch.from_numpy(state).double().to(device).unsqueeze(0)
    with torch.no_grad():
        alpha, beta = net(state)[0]
    dist = Beta(alpha, beta) # Beta distribution make the samples between [0, 1]
    #print(alpha, beta) 
    action = dist.sample()
    action_logp = dist.log_prob(action).sum(dim=1)

    action = action.squeeze().cpu().numpy()
    action_logp = action_logp.item()
    return action, action_logp

max_grad_norm = 0.5
clip_param = 0.1  # epsilon in clipped loss
ppo_epoch = 10

def update(buffer, net, optimizer, batch_size):

    s = torch.tensor(buffer['s'], dtype=torch.double).to(device)
    a = torch.tensor(buffer['a'], dtype=torch.double).to(device)
    r = torch.tensor(buffer['r'], dtype=torch.double).to(device).view(-1, 1)
    s_next = torch.tensor(buffer['s_next'], dtype=torch.double).to(device)

    old_action_logp = torch.tensor(buffer['action_logp'], dtype=torch.double).to(device).view(-1, 1)

    with torch.no_grad():
        target_value = r + args.gamma * net(s_next)[1]
        adv = target_value - net(s)[1]

    for _ in range(ppo_epoch):
        for index in BatchSampler(SubsetRandomSampler(range(len(buffer))), batch_size, False):

            alpha, beta = net(s[index])[0]
            dist = Beta(alpha, beta)
            action_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
            ratio = torch.exp(action_logp - old_action_logp[index]) 

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(net(s[index])[1], target_value[index]) #SmoothL1Loss
            loss = action_loss + 2. * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def rgb2gray(rgb, norm=True):
    # convert rgb image to gray
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114]) # the formula of convert rgb2gray
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.1)

def rwd_memory():
    # record reward last 100 steps reward
    count = 0
    length = 100
    history = np.zeros(length)

    def memory(rwd):
        nonlocal count
        history[count] = rwd
        count = (count + 1) % length
        return np.mean(history)

    return memory

def save_param(net):
    torch.save(net.state_dict(), 'param/ppo_net_params.pkl')
    torch.save(net.state_dict(), 'param/ppo_net_params.pt')


if __name__ == "__main__":
    agent = Car_Agent()
    env = New_Env()
    net = PPO_Net().double().to(device) 
    optimizer = optim.Adam(net.parameters(), lr=1e-3) 
    batch_size = 128
    training_records = []
    running_score = 0

    for i_ep in range(3000): # episode
        score = 0
        state = env.reset()

        for t in range(10000): # max steps in each run
            action, action_logp = select_action(net, state)
            #print(action)
            state_, rwd, done, die = env.step_single_image(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store_transition((state, action, action_logp, rwd, state_)):
                print('updating')
                update(agent.getBuffer(), net, optimizer, batch_size)
            score = score + rwd
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:

            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            save_param(net)
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
