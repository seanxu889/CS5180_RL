'''
testing PPO algorithm for CarRacing-v0 gym environment
'''

import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import imageio 
import os
from utils_dagger import str2bool

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=1, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment') 

# outputs the expert data, training the DAGGER if necessary
parser.add_argument("--out_dir", help="directory in which to save the expert's data", default='./expert_dataset/train')
parser.add_argument("--save_expert_actions", type=str2bool, help="save the images and expert actions in the training set",
                        default=False)

args = parser.parse_args()

device = torch.device("cpu")


class New_Env():
    """
    Modified Environment 
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset_get_rgb(self): 
        img_rgb = self.env.reset()
        return img_rgb

    
    def reset(self):
        self.av_r = reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack
        return np.array(self.stack)
    
    def step_single_image(self, action):
        total_rwd = 0
        for i in range(args.action_repeat):
            img_rgb, rwd, die, _ = self.env.step(action)
            
            # don't penalize "die state"
            if die:
                rwd += 100
            
            # penalty grassland area (green)
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                rwd -= 0.05
            total_rwd += rwd
            
            # end episode if no reward received
            done = True if self.av_r(rwd) <= -0.1 else False
            if done or die:
                break
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return img_rgb, total_rwd, done, die
    
    
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
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
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

        

def select_action(net, state):
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)
    with torch.no_grad():
        alpha, beta = net(state)[0]
    action = alpha / (alpha + beta)

    action = action.squeeze().cpu().numpy()
    return action

def load_param():
    # return torch.load('param/ppo_net_params.pkl')
    return torch.load('./20191130_results/param/ppo_net_params.pkl')


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
    count = 0
    length = 100
    history = np.zeros(length)

    def memory(reward):
        nonlocal count
        history[count] = reward
        count = (count + 1) % length
        return np.mean(history)

    return memory

if __name__ == "__main__":
    param = load_param()
    env = New_Env()
    net = PPO_Net().float().to(device)
    net.load_state_dict(param)
    training_records = []
    running_score = 0
    state = env.reset() 
    state_rgb = env.reset_get_rgb() 
    for i_ep in range(10):
        score = 0
        state = env.reset()
        state_rgb = env.reset_get_rgb()

        for t in range(1000):
            action = select_action(net, state) 
            action_to_gym = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]) 

            state_rgb_next, _, __, ___ = env.step_single_image(action_to_gym) 
            state_, reward, done, die = env.step(action_to_gym)  
     
            ### need to add args.save_expert_actions, args.out_dir 
            ## import imageio            
            
            if True: 
            #state contains four images, need to unpack it to four different images
            #if args.save_expert_actions: 
                imageio.imwrite(os.path.join(args.out_dir, 'expert_%f_%f_%f.jpg' % (action_to_gym[0], action_to_gym[1], action_to_gym[2])), state_rgb)
                
            if args.render:
                env.render()
            score += reward
            state = state_  
            state_rgb = state_rgb_next
            ## added 
            #prev_ac = action_to_gym
            
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
