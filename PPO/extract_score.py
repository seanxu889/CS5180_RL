#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:23:46 2019

@author: seanxu

extract information from terminal results

"""

import numpy as np
import re
import matplotlib.pyplot as plt

file = open('./20191130_results/20191130_carracing_terminal_copy.txt', 'r')
file.seek(0)
all_lines = file.readlines()
#print(all_lines)

score_lines = []
for i in range(len(all_lines)):
    if all_lines[i].startswith('Ep ') & all_lines[i].endswith('\n'):
        score_lines.append(all_lines[i])

m_ave = []
for j in range(len(score_lines)):
    m_ave.append(re.findall(r"average score: (.+?)\n",score_lines[j]))

for i in range(len(m_ave)):
    m_ave[i][0] = float(m_ave[i][0])
    

m_last = []
for j in range(len(score_lines)):
    m_last.append(re.findall(r"Last score: (.+?)\tMoving",score_lines[j]))

for i in range(len(m_last)):
    m_last[i][0] = float(m_last[i][0])

ep_num = np.linspace(0, 10*(len(m_ave)-1), (len(m_ave)))
ep_num.reshape(213)
    
plt.plot(ep_num, m_ave, 'r', label = 'running score')
plt.plot(ep_num, m_last, label = 'score')

plt.title('PPO car racing result')
plt.ylabel('score')
plt.xlabel('episode')
plt.legend()
plt.savefig('./20191130_results/score.jpg', dpi=300)