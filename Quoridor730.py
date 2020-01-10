import numpy as np
import time
import copy
import os
import random
from asyncio import Future
import asyncio
from asyncio.queues import Queue
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from collections import deque, namedtuple
from policy_value_network import policy_value_network 

# import mcts

BOARD_ROWS = 7
BOARD_COLS = 7
BOARD_SIZE = BOARD_COLS * BOARD_ROWS
FENCE_SIZE = (BOARD_ROWS - 1) * BOARD_COLS
DIRECTIONS = {
    'N': 0, 'E': 1, 'NN': 2, 'EE': 3,
    'NE': 4, 'NW': 5, 'SE': 6, 'SW': 7,
    'WW': 8, 'SS': 9, 'W': 10, 'S': 11
}
N_SPACE_FENCE = (BOARD_ROWS - 1) * (BOARD_COLS - 1)
ACTION_SPACE = 84
#动作空间组成：0~11棋子动作，12~47横挡板动作，48~83竖挡板动作

class point:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.basecost = 0
        self.parent = None
        self.cost = 0

def valid_pawn_act_without_op(state, my_pos_x, my_pos_y):
        """获取合法走棋动作，不考虑对手位置，用在搜索路径的时候
        """
        valid = []

        # 分类讨论
        if my_pos_y > 0 and state.hor_fence_pos[my_pos_y-1, my_pos_x] == 0:     #没在顶部边界且北边没有障碍
            valid.append(DIRECTIONS['N'])   #那么可以往北走一步
                
        #其他方向同理
        if my_pos_y < BOARD_ROWS-1 and state.hor_fence_pos[my_pos_y, my_pos_x] == 0:    #South
            valid.append(DIRECTIONS['S'])

        if my_pos_x < BOARD_COLS-1 and state.ver_fence_pos[my_pos_y, my_pos_x] == 0:    #East
            valid.append(DIRECTIONS['E'])

        if my_pos_x > 0 and state.ver_fence_pos[my_pos_y, my_pos_x-1] == 0:             #West
            valid.append(DIRECTIONS['W'])

        return valid

class a_star:
    def __init__(self, state, start_p, target_row):
        self.open_set = set()
        self.close_set = set()
        self.start_p = start_p
        self.state = state
        self.target_row = target_row

    def get_baseCost(self, p):
        if p.parent == None:
            return 0
        else:
            p.basecost = p.parent.basecost + 1
            return p.basecost
    
    def get_totalCost(self, p):
        return self.get_baseCost(p) + abs(self.target_row - p.row)

    def isInOpenSet(self, p):
        for _p in self.open_set:
            if p.row == _p.row and p.col == _p.col:
                return True
        return False

    def isInCloseSet(self, p):
        for _p in self.close_set:
            if p.row == _p.row and p.col == _p.col:
                return True
        return False

    def isEnd(self, p):
        return p.row == self.target_row

    def processPoint(self, row, col, parent):
        p = point(row, col)     
        if self.isInCloseSet(p):
            return
        if not self.isInOpenSet(p):
            p.parent = parent
            p.cost = self.get_totalCost(p)
            self.open_set.add(p)
    
    def selectOpenPoint(self):
        return min(self.open_set, key = lambda p: p.cost)
    
    def firstStep(self, p):
        if p == self.start_p:
            return p
        while not p.parent == self.start_p:
            p = p.parent
        return p

    def run(self):
        self.open_set.add(self.start_p)
        while True:
            if len(self.open_set) > 0:
                p = self.selectOpenPoint()
            else:
                return [-1, -1] #false, no path found
            
            if self.isEnd(p):
                p = self.firstStep(p)
                return [p.row, p.col]

            self.open_set.discard(p)
            self.close_set.add(p)

            valid_pawn_act = valid_pawn_act_without_op(self.state, p.col, p.row)

            for act in valid_pawn_act:
                if act == DIRECTIONS['N']:
                    self.processPoint(p.row-1, p.col, p)
                elif act == DIRECTIONS['S']:
                    self.processPoint(p.row+1, p.col, p)
                elif act == DIRECTIONS['E']:
                    self.processPoint(p.row, p.col+1, p)
                elif act == DIRECTIONS['W']:
                    self.processPoint(p.row, p.col-1, p)

class State:

    def __init__(self):
        self.pawn_pos = [{'y':6, 'x':3}, {'y':0, 'x':3}]                    #棋子位置
        self.ver_fence_pos = np.zeros((BOARD_ROWS, BOARD_COLS-1))           #横竖障碍位置棋盘，是7*6,
        self.hor_fence_pos = np.zeros((BOARD_ROWS-1, BOARD_COLS))
        self.winner = None
        self.fence_num = [8, 8]                                             #双方剩余障碍数
        # self.end = False
    
    def Clone(self):                                                        #拷贝一个State，在树搜索时有用
        st = State()
        st.pawn_pos = copy.deepcopy(self.pawn_pos)
        st.ver_fence_pos = copy.deepcopy(self.ver_fence_pos)
        st.hor_fence_pos = copy.deepcopy(self.hor_fence_pos)
        st.fence_num = self.fence_num.copy()
        # st.pawn_pos[0] = self.pawn_pos[0].copy()
        # st.pawn_pos[1] = self.pawn_pos[1].copy()
        # st.ver_fence_pos = self.ver_fence_pos.copy()
        # st.hor_fence_pos = self.hor_fence_pos.copy()
        return st

    def is_end(self):

        if self.pawn_pos[1]['y']  == 6:
            self.winner = 1
            return True
        if self.pawn_pos[0]['y'] == 0:
            self.winner = 0
            return True

        return False

    def print_state(self):                                  #在终端打印盘面
        
        print('------------BOARD------------')
        # print('-----------------------------')
        for j in range(BOARD_ROWS):
            out = '| '
            for i in range(BOARD_COLS):
                if self.pawn_pos[0]['x'] == i and self.pawn_pos[0]['y'] == j:
                    token = 'X'
                elif self.pawn_pos[1]['x'] == i and self.pawn_pos[1]['y'] == j:
                    token = 'O'
                else:
                    token = '-'
                if i < BOARD_COLS - 1:
                    if self.ver_fence_pos[j, i] != 0:
                        out += token + ' █ '
                        # out += token + ' ' + str(int(self.ver_fence_pos[j, i])) + ' '
                    else:
                        out += token + ' | '
                else:
                    out += token + ' | '
            print(out)
            out = '-'
            for i in range(BOARD_COLS):
                if j < BOARD_ROWS - 1:
                    if self.hor_fence_pos[j, i] != 0:
                        out += '▄▄▄'
                        # out += ' ' + str(int(self.hor_fence_pos[j, i])) + ' '
                    else:
                        out += '---'
                else:
                    out += '---'
                out += '-'
            print(out)
        print('')
    
    
class Judger:                               #裁判类，版本多次更迭后有一点没用。。主要是负责双方交替

    def __init__(self, player1, player2, first_player):
        self.player = [player1, player2]
        self.first_player = first_player
        # self.current_state = State()

    # def reset(self):
    #     self.p1.reset()
    #     self.p2.reset()

    def alternate(self):
        while True:
            yield self.player[self.first_player]
            yield self.player[1-self.first_player]


class Player:           #玩家类

    def __init__(self,symbol = 0):
        self.symbol = symbol                                        #玩家号，0和1
        if self.symbol == 0:
            # self.pawn_pos = {'x':3, 'y':6}
            self.taget_row = 0                                      #目标行，到达此行即胜利
        else:
            # self.pawn_pos = {'x':3, 'y':0}
            self.taget_row = BOARD_ROWS-1
        # self.search_map = np.zeros((BOARD_ROWS, BOARD_COLS))        #递归搜索路径是否合法时用的

    def valid_pawn_actions(self, state, withoutAstar):
        """获取本玩家所有合法走子动作
        """
        valid = []

        my_pos_x = state.pawn_pos[self.symbol]['x']
        my_pos_y = state.pawn_pos[self.symbol]['y']
        op_pos_x = state.pawn_pos[1-self.symbol]['x']
        op_pos_y = state.pawn_pos[1-self.symbol]['y']

        # 判断对面棋子是否相邻
        opponent_north = my_pos_y - 1 == op_pos_y and my_pos_x == op_pos_x
        opponent_south = my_pos_y + 1 == op_pos_y and my_pos_x == op_pos_x
        opponent_east = my_pos_x + 1 == op_pos_x and my_pos_y == op_pos_y
        opponent_west = my_pos_x - 1 == op_pos_x and my_pos_y == op_pos_y
        # 分类讨论
        if my_pos_y > 0 and state.hor_fence_pos[my_pos_y-1, my_pos_x] == 0:     #没在顶部边界且北边没有障碍
            if not opponent_north:              #北边没有棋子相邻
                if not withoutAstar:
                    p = point(my_pos_y-1, my_pos_x)                     #A*搜索最短路径，路径第一步不是当前位置。否则就是走远了或者是前方死路
                    astar = a_star(state, p, self.taget_row)            #从将要走的那一步开始搜索
                    ret = astar.run()
                    if not (ret[0] == my_pos_y and ret[1] == my_pos_x):
                        valid.append(DIRECTIONS['N'])                   #那么可以往北走一步
                else:
                    valid.append(DIRECTIONS['N'])
            else:                               #如果北边有棋子相邻
                if op_pos_y > 0 and state.hor_fence_pos[op_pos_y-1, op_pos_x] == 0:    #对手北边不是边界和障碍
                    valid.append(DIRECTIONS['NN'])  #可以越过对手往北走一步
                if op_pos_x < BOARD_COLS-1 and state.ver_fence_pos[op_pos_y, op_pos_x] == 0:  #对手东边
                    valid.append(DIRECTIONS['NE'])  #越过对手，向北再向东
                if op_pos_x > 0 and state.ver_fence_pos[op_pos_y, op_pos_x-1] == 0:       #对手西边
                    valid.append(DIRECTIONS['NW'])  #向北再向西
                
        #其他方向同理
        if my_pos_y < BOARD_ROWS-1 and state.hor_fence_pos[my_pos_y, my_pos_x] == 0:    #South
            if not opponent_south:
                if not withoutAstar:
                    p = point(my_pos_y+1, my_pos_x)
                    astar = a_star(state, p, self.taget_row)
                    ret = astar.run()
                    if not (ret[0] == my_pos_y and ret[1] == my_pos_x):
                        valid.append(DIRECTIONS['S'])
                else:
                    valid.append(DIRECTIONS['S'])
            else:
                if op_pos_y < BOARD_ROWS-1 and state.hor_fence_pos[op_pos_y, op_pos_x] == 0:
                    valid.append(DIRECTIONS['SS'])
                if op_pos_x < BOARD_COLS-1 and state.ver_fence_pos[op_pos_y, op_pos_x] == 0:
                    valid.append(DIRECTIONS['SE'])
                if op_pos_x > 0 and state.ver_fence_pos[op_pos_y, op_pos_x-1] == 0:
                    valid.append(DIRECTIONS['SW'])

        if my_pos_x < BOARD_COLS-1 and state.ver_fence_pos[my_pos_y, my_pos_x] == 0:    #East
            if not opponent_east:
                if not withoutAstar:
                    p = point(my_pos_y, my_pos_x+1)
                    astar = a_star(state, p, self.taget_row)
                    ret = astar.run()
                    if not (ret[0] == my_pos_y and ret[1] == my_pos_x):
                        valid.append(DIRECTIONS['E'])
                else:
                    valid.append(DIRECTIONS['E'])
            else:
                if op_pos_x < BOARD_COLS-1 and state.ver_fence_pos[op_pos_y, op_pos_x] == 0:
                    valid.append(DIRECTIONS['EE'])
                if op_pos_y > 0 and state.hor_fence_pos[op_pos_y-1, op_pos_x] == 0:
                    valid.append(DIRECTIONS['NE'])
                if op_pos_y < BOARD_ROWS-1 and state.hor_fence_pos[op_pos_y, op_pos_x] == 0:
                    valid.append(DIRECTIONS['SE'])

        if my_pos_x > 0 and state.ver_fence_pos[my_pos_y, my_pos_x-1] == 0:             #West
            if not opponent_west:
                if not withoutAstar:
                    p = point(my_pos_y, my_pos_x-1)
                    astar = a_star(state, p, self.taget_row)
                    ret = astar.run()
                    if not (ret[0] == my_pos_y and ret[1] == my_pos_x):
                        valid.append(DIRECTIONS['W'])
                else:
                    valid.append(DIRECTIONS['W'])
            else:
                if op_pos_x > 0 and state.ver_fence_pos[op_pos_y, op_pos_x-1] == 0:
                    valid.append(DIRECTIONS['WW'])
                if op_pos_y > 0 and state.hor_fence_pos[op_pos_y-1, op_pos_x] == 0:
                    valid.append(DIRECTIONS['NW'])
                if op_pos_y < BOARD_ROWS-1 and state.hor_fence_pos[op_pos_y, op_pos_x] == 0:
                    valid.append(DIRECTIONS['SW'])

        return valid

    # def check_path(self, target_row, hor_fence_pos, ver_fence_pos, pos_x, pos_y, search_map):
    #     """ 递归搜索路径，检查是否能达到指定行。每次往上下左右走一步，到达指定行返回True
    #         走过的格子在map标记，避免重复递归
    #     """
    #     if pos_y == target_row:
    #         return True

    #     valid_dirs = valid_pawn_act_without_op(hor_fence_pos, ver_fence_pos, pos_x, pos_y)
    #     for direction in valid_dirs:
    #         if direction == DIRECTIONS['N'] and search_map[pos_y-1, pos_x] == 0:
    #             search_map[pos_y-1, pos_x] = 1
    #             if self.check_path(target_row, hor_fence_pos, ver_fence_pos, pos_x, pos_y-1, search_map):
    #                 return True
    #         if direction == DIRECTIONS['S'] and search_map[pos_y+1, pos_x] == 0:
    #             search_map[pos_y+1, pos_x] = 1
    #             if self.check_path(target_row, hor_fence_pos, ver_fence_pos, pos_x, pos_y+1, search_map):
    #                 return True
    #         if direction == DIRECTIONS['E'] and search_map[pos_y, pos_x+1] == 0:
    #             search_map[pos_y, pos_x+1] = 1
    #             if self.check_path(target_row, hor_fence_pos, ver_fence_pos, pos_x+1, pos_y, search_map):
    #                 return True
    #         if direction == DIRECTIONS['W'] and search_map[pos_y, pos_x-1] == 0:
    #             search_map[pos_y, pos_x-1] = 1
    #             if self.check_path(target_row, hor_fence_pos, ver_fence_pos, pos_x-1, pos_y, search_map):
    #                 return True
            
    #     return False

    # def valid_fence_actions(self, state):
    #     """获取合法障碍摆放动作
    #     """

    #     my_pos_x = state.pawn_pos[self.symbol]['x']
    #     my_pos_y = state.pawn_pos[self.symbol]['y']
    #     op_pos_x = state.pawn_pos[1-self.symbol]['x']
    #     op_pos_y = state.pawn_pos[1-self.symbol]['y']

    #     valid = []
    #     search_map = np.zeros((BOARD_ROWS, BOARD_COLS))
            
    #     for j in range(BOARD_ROWS-1):
    #         for i in range(BOARD_COLS-1):
    #             if state.hor_fence_pos[j, i] == 0 and state.hor_fence_pos[j, i+1] == 0 \
    #             and not (state.ver_fence_pos[j, i] == 1 and state.ver_fence_pos[j+1, i] == 2):  #有连续两个相邻的空位，且没有另一方向的障碍
    #                 hor_fence_temp = state.hor_fence_pos.copy()                                 #可以放置，则假设在这里放障碍
    #                 hor_fence_temp[j, i] = 1
    #                 hor_fence_temp[j, i+1] = 2
    #                 if self.check_path(self.taget_row, hor_fence_temp, state.ver_fence_pos, my_pos_x, my_pos_y, search_map):
    #                     search_map = np.zeros((BOARD_ROWS, BOARD_COLS))
    #                     if self.check_path(BOARD_ROWS-1-self.taget_row, hor_fence_temp, state.ver_fence_pos, op_pos_x, op_pos_y, search_map):  #检查是否有通路
    #                         valid.append(j*(BOARD_COLS-1)+i + 12)   #前12个动作为棋子动作。
    #                                                                 #因为一个障碍由两个组成，这里只考虑挡板1，可用挡板位置为6*6
    #                         # print('valid_hor:', j, i)
    #                         # print(self.search_map)
    #                 search_map = np.zeros((BOARD_ROWS, BOARD_COLS))

    #     for j in range(BOARD_ROWS-1):
    #         for i in range(BOARD_COLS-1):
    #             if state.ver_fence_pos[j, i] == 0 and state.ver_fence_pos[j+1, i] == 0 \
    #             and not (state.hor_fence_pos[j, i] == 1 and state.hor_fence_pos[j, i+1] == 2):
    #                 ver_fence_temp = state.ver_fence_pos.copy()
    #                 ver_fence_temp[j, i] = 1
    #                 ver_fence_temp[j+1, i] = 2
    #                 if self.check_path(self.taget_row, state.hor_fence_pos, ver_fence_temp, my_pos_x, my_pos_y, search_map):
    #                     search_map = np.zeros((BOARD_ROWS, BOARD_COLS))
    #                     if self.check_path(BOARD_ROWS-1-self.taget_row, state.hor_fence_pos, ver_fence_temp, op_pos_x, op_pos_y, search_map):
    #                         valid.append(j*(BOARD_COLS-1)+i + 12 + N_SPACE_FENCE)  #12个棋子动作，36个横挡板动作
    #                         # print('ver:', j, i)
    #                 search_map = np.zeros((BOARD_ROWS, BOARD_COLS))

    #     return valid

    def valid_fence_actions(self, state):
        """获取合法障碍摆放动作
        """

        my_pos_x = state.pawn_pos[self.symbol]['x']
        my_pos_y = state.pawn_pos[self.symbol]['y']
        op_pos_x = state.pawn_pos[1-self.symbol]['x']
        op_pos_y = state.pawn_pos[1-self.symbol]['y']

        valid = []
            
        for j in range(BOARD_ROWS-1):
            for i in range(BOARD_COLS-1):
                if state.hor_fence_pos[j, i] == 0 and state.hor_fence_pos[j, i+1] == 0 \
                and not (state.ver_fence_pos[j, i] == 1 and state.ver_fence_pos[j+1, i] == 2):  #有连续两个相邻的空位，且没有另一方向的障碍
                    state.hor_fence_pos[j, i] = 1               #则假设在此放障碍
                    state.hor_fence_pos[j, i+1] = 2
                    astar = a_star(state, point(my_pos_y, my_pos_x), self.taget_row)
                    ret1 = astar.run()
                    astar = a_star(state, point(op_pos_y, op_pos_x), BOARD_ROWS-1-self.taget_row)
                    ret2 = astar.run()
                    if ret1[0] != -1 and ret2[0] != -1:
                        valid.append(j*(BOARD_COLS-1)+i + 12)   #前12个动作为棋子动作。

                    state.hor_fence_pos[j, i] = 0               #不管行不行，要把状态改回去
                    state.hor_fence_pos[j, i+1] = 0

        for j in range(BOARD_ROWS-1):
            for i in range(BOARD_COLS-1):
                if state.ver_fence_pos[j, i] == 0 and state.ver_fence_pos[j+1, i] == 0 \
                and not (state.hor_fence_pos[j, i] == 1 and state.hor_fence_pos[j, i+1] == 2):
                    state.ver_fence_pos[j, i] = 1               #则假设在此放障碍
                    state.ver_fence_pos[j+1, i] = 2
                    astar = a_star(state, point(my_pos_y, my_pos_x), self.taget_row)
                    ret1 = astar.run()
                    astar = a_star(state, point(op_pos_y, op_pos_x), BOARD_ROWS-1-self.taget_row)
                    ret2 = astar.run()
                    if ret1[0] != -1 and ret2[0] != -1:
                        valid.append(j*(BOARD_COLS-1)+i + 12 + N_SPACE_FENCE)  #12个棋子动作，36个横挡板动作
                    state.ver_fence_pos[j, i] = 0               #不管行不行，要把状态改回去
                    state.ver_fence_pos[j+1, i] = 0

        return valid

    def get_actions(self, state):
        """ 获取所有合法动作列表
        """
        op_pos_x = state.pawn_pos[1-self.symbol]['x']
        op_pos_y = state.pawn_pos[1-self.symbol]['y']

        # pawn_valid = self.valid_pawn_actions(state, withoutAstar = False)
        # if pawn_valid == []:
        pawn_valid = self.valid_pawn_actions(state, withoutAstar = True)
        fence_valid = []
        if state.fence_num[self.symbol] > 0:
            fence_valid = self.valid_fence_actions(state)
        valid = pawn_valid + fence_valid
        return valid
    
    def handle_pawn_aciton(self, action, state):
        """处理棋子动作，改变state中对应棋子位置
        """
        if action == DIRECTIONS['N']:
            state.pawn_pos[self.symbol]['y'] -= 1
        elif action == DIRECTIONS['S']:
            state.pawn_pos[self.symbol]['y'] += 1
        elif action == DIRECTIONS['E']:
            state.pawn_pos[self.symbol]['x'] += 1
        elif action == DIRECTIONS['W']:
            state.pawn_pos[self.symbol]['x'] -= 1
        elif action == DIRECTIONS['NN']:
            state.pawn_pos[self.symbol]['y'] -= 2
        elif action == DIRECTIONS['SS']:
            state.pawn_pos[self.symbol]['y'] += 2
        elif action == DIRECTIONS['EE']:
            state.pawn_pos[self.symbol]['x'] += 2
        elif action == DIRECTIONS['WW']:
            state.pawn_pos[self.symbol]['x'] -= 2
        elif action == DIRECTIONS['NW']:
            state.pawn_pos[self.symbol]['y'] -= 1
            state.pawn_pos[self.symbol]['x'] -= 1
        elif action == DIRECTIONS['NE']:
            state.pawn_pos[self.symbol]['y'] -= 1
            state.pawn_pos[self.symbol]['x'] += 1
        elif action == DIRECTIONS['SW']:
            state.pawn_pos[self.symbol]['y'] += 1
            state.pawn_pos[self.symbol]['x'] -= 1
        elif action == DIRECTIONS['SE']:
            state.pawn_pos[self.symbol]['y'] += 1
            state.pawn_pos[self.symbol]['x'] += 1
        else:
            print('error!')
    
    def do_action(self, action, state):
        """本玩家进行动作，改变state。前12是走子动作，12-48、48-84分别是横竖障碍动作
        """

        if action < 12:
            self.handle_pawn_aciton(action, state)
        elif action >= 12 and action < 12 + N_SPACE_FENCE:
            row_t = (action-12) // (BOARD_COLS-1)
            col_t = (action-12) % (BOARD_COLS-1)
            state.hor_fence_pos[row_t, col_t] = 1
            state.hor_fence_pos[row_t, col_t+1] = 2
            state.fence_num[self.symbol] -= 1
        elif action >= 12 + N_SPACE_FENCE and action < 12 + N_SPACE_FENCE*2:
            row_t = (action-48) // (BOARD_COLS-1)
            col_t = (action-48) % (BOARD_COLS-1)
            state.ver_fence_pos[row_t, col_t] = 1
            state.ver_fence_pos[row_t+1, col_t] = 2
            state.fence_num[self.symbol] -= 1


class Node:
    """ A node in the game tree. 
    """
    def __init__(self, action = None, parent = None, current_player = None, prior_p = None):
        self.action = action # the move that got us to this node 这个动作是上个结点做的，以达到当前结点
        self.parentNode = parent # "None" for the root node
        self.childNodes = []    #子结点，也可以以字典的方式写，写成动作-子结点的映射形式
        self.visit = 1e-10  #该结点访问次数，每次更新+1。防止除0warning
        self.W = 0          #total value
        self.Q = 0          #可以理解该结点为平均价值，用于评价该结点
        self.depth = 0      #树深度，步数过多时强制终局
        self.prior_p = prior_p  #神经网络输出的先验概率
        self.current_player = current_player    #这个结点将要做动作的玩家，不是做self.action的玩家（ps可能改成刚刚行动的玩家比较好理解，和action统一）
        
    def SelectChild(self, c_puct=5):
        # U+Q
        s = max(self.childNodes, key = lambda c: c_puct * c.prior_p * np.sqrt(self.visit)/(1+c.visit) + c.Q)
        s.depth = self.depth + 1    #深度+1，为了限制递归深度的
        return s
    
    def AddChild(self, act, p, prior_p):
        n = Node(action = act, parent = self, current_player = p, prior_p = prior_p)
        self.childNodes.append(n)
        return n
    
    def Update(self, value):
        """ Update this node,访问次数+1，更新平均价值Q
        """
        self.visit += 1 
        self.W += value
        self.Q = self.W / self.visit
        
        # self.Q += 1.0 * (value - self.Q) / self.visit
        
def softmax(x):
    """归一化到0-1，和为1，主要用于概率
    """
    # print(x)
    probs = np.exp(x - np.max(x))
    # print(np.sum(probs))
    probs /= np.sum(probs)
    return probs

QueueItem = namedtuple("QueueItem", "feature future")   #多线程模拟时用到的，队列处理网络

class mcts_tree:
    def __init__(self, network, first_player, search_threads=16):
        self.now_expanding = set()                      #正在进行计算的结点集合，防止多线程冲突
        self.expanded = set()                           #已经被扩展过的结点，用于判断
        self.network = network                          #策略价值网络
        self.running_simulation_num = 0
        self.sem = asyncio.Semaphore(search_threads)    #异步线程信号，线程默认16
        self.queue = Queue(search_threads)              #神经网络多线程处理的队列
        self.loop = asyncio.get_event_loop()            #处理异步用的，这个模块还不是很理解
        self.virtual_loss = 3                           #虚拟损失，避免多个线程选择同一个结点
        self.rootNode = Node(current_player = first_player) #根结点，在一局棋开始时创建，在每一步结束后更新，复用子树

    async def push_queue(self, features):   #异步将盘面特征，也就是神经网络的输入，加入队列
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    async def start_runout(self, node, judger, state):  #开始模拟
        self.running_simulation_num += 1

        with await self.sem:
            value = await self.runout(node, judger, state)
            self.running_simulation_num -= 1
            return value


    async def runout(self, node, judger, state):
        """ 递归直到游戏结束
        """
        now_expanding = self.now_expanding

        while node in now_expanding:        #如果该结点正在被其他线程处理，就等一会儿
            await asyncio.sleep(1e-4)

        if not self.is_expanded(node):
            """is leaf node try evaluate and expand"""
            self.now_expanding.add(node)
            net_inputs = get_net_inputs(state, node.current_player)     #将棋盘转换为神经网络输入的特征

            future = await self.push_queue(net_inputs)  # type: Future
            await future
            action_probs, value = future.result()       #其实就是network.forword(),这里用了队列处理
            legal_actions = judger.player[node.current_player].get_actions(state)   #获取所有合法状态
            # tot_p = 1e-8                        #总概率，后面求
            if node.current_player == 1:
                for action in legal_actions:    #获取合法动作对应的概率，扩展子结点
                    node.AddChild(action, judger.player[1 - node.current_player].symbol, action_probs[action_flip(action)])
                    # tot_p += action_probs[action_flip(action)]      #玩家1需要翻转棋盘，都以玩家0的视角去处理
            else:
                for action in legal_actions:
                    node.AddChild(action, judger.player[1 - node.current_player].symbol, action_probs[action])
                    # tot_p += action_probs[action]
            
            # for childnod in node.childNodes:
            #     childnod.prior_p /= tot_p       

            self.expanded.add(node)

            self.now_expanding.remove(node)

            return value[0] * -1        #这一结点的价值×-1返回上一层
        
        else:
            """node has already expanded. Enter select phase."""
            node = node.SelectChild()       #根据U+Q选择一个子结点

            node.visit += self.virtual_loss
            node.W -= self.virtual_loss
            node.Q = node.W / node.visit

            judger.player[1-node.current_player].do_action(node.action, state)  #更新状态，到当前结点

            if state.is_end() == True:
                if state.winner == 1-node.current_player:   #应该是这样：上一结点的玩家赢了，说明当前state是有价值的，这个state是由上一个node的玩家下的
                    value = 1
                elif state.winner == node.current_player:       #这个想法是错的：自己赢了就是1，返回给上一步-1
                    value = -1
                else:
                    value = 0
            elif node.depth > 100:      #递归深度大于一定值，没有结果即强制平局
                value = 0
            else:
                value = await self.runout(node, judger, state)      #还没结束，继续递归

            node.visit -= self.virtual_loss
            node.W += self.virtual_loss
            node.Update(value)
            return value * -1

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            features = np.asarray([item.feature for item in item_list])    # asarray

            action_probs, value = self.network.forward(features)
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    def is_expanded(self, key) -> bool:
        """Check expanded status"""
        return key in self.expanded

    def zero_mcts(self, rootstate, judger, itermax, selfplay =False, temperature = 1e-3):
        """ 蒙特卡洛树主函数，模拟一定次数后，返回选择的落子
        """
        node = self.rootNode
        state = rootstate.Clone()   #拷贝状态，避免模拟改变实际棋盘
        if not self.is_expanded(node):
            net_inputs = get_net_inputs(state, node.current_player)
            net_inputs = np.expand_dims(net_inputs, 0)
            action_probs, value = self.network.forward(net_inputs)
            action_probs = action_probs.flatten()

            legal_actions = judger.player[node.current_player].get_actions(state)
            # tot_p = 1e-8
            if node.current_player == 1:
                for action in legal_actions:
                    node.AddChild(action, judger.player[1 - node.current_player].symbol, action_probs[action_flip(action)])
                    # tot_p += action_probs[action_flip(action)]
            else:
                for action in legal_actions:
                    node.AddChild(action, judger.player[1 - node.current_player].symbol, action_probs[action])
            #         tot_p += action_probs[action]
            
            # for childnod in node.childNodes:
            #     childnod.prior_p /= tot_p

            self.expanded.add(node)

        coroutine_list = []
        for _ in range(itermax):
            coroutine_list.append(self.start_runout(node, judger, state.Clone()))   #clone!!!
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        actions = []
        visits =[]
        move_probs = {}
        for nod in self.rootNode.childNodes:
            actions.append(nod.action)
            visits.append(nod.visit)
        probs = softmax(1.0 / temperature * np.log(visits))

        for i in range(len(actions)):
            move_probs[actions[i]] = probs[i]
        if selfplay:
            act = np.random.choice(actions, p=0.75 * probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        self.expanded.discard(self.rootNode)        #复用子树,去掉根节点，往下走一步
        for nod in self.rootNode.childNodes:
            if nod.action == act:
                win_rate = nod.Q
                self.rootNode = nod
                break
        self.rootNode.parentNode = None
        self.rootNode.depth = 0

        return act, move_probs, win_rate

def get_net_inputs(state, current_player):
    #输入有6层，0：横障碍位置，1：竖障碍位置，2：我方棋子位置，3：对手棋子位置，4：我方剩余障碍数，5：对方剩余障碍数
    #障碍数 8：全图为1，1-7：对应行为1, 0：全图0
    net_inputs = np.zeros(shape = (BOARD_ROWS, BOARD_COLS, 6), dtype = np.float32)
    
    if current_player == 0:
        p0_y = state.pawn_pos[0]['y']
        p0_x = state.pawn_pos[0]['x']
        p1_y = state.pawn_pos[1]['y']
        p1_x = state.pawn_pos[1]['x']
        net_inputs[p0_y][p0_x][2] = 1
        net_inputs[p1_y][p1_x][3] = 1

        if state.fence_num[0] == 8:
            for j in range(BOARD_ROWS):
                for i in range(BOARD_COLS):
                    net_inputs[j][i][4] = 1
        elif state.fence_num[0] > 0:
            for j in range(BOARD_ROWS):
                net_inputs[j][state.fence_num[0]-1][4] = 1

        if state.fence_num[1] == 8:
            for j in range(BOARD_ROWS):
                for i in range(BOARD_COLS):
                    net_inputs[j][i][5] = 1
        elif state.fence_num[1] > 0:
            for j in range(BOARD_ROWS):
                net_inputs[j][state.fence_num[1]-1][5] = 1

        for j in range(BOARD_ROWS-1):
            for i in range(BOARD_COLS-1):
                if state.hor_fence_pos[j, i] == 1:
                    net_inputs[j][i][0] = 1
                    net_inputs[j][i+1][0] = 1
                if state.ver_fence_pos[j, i] == 1:
                    net_inputs[j][i][1] = 1
                    net_inputs[j+1][i][1] = 1

    else:   #翻转棋盘
        p0_y = BOARD_ROWS - 1 - state.pawn_pos[0]['y']
        p0_x = BOARD_COLS - 1 - state.pawn_pos[0]['x']
        p1_y = BOARD_ROWS - 1 - state.pawn_pos[1]['y']
        p1_x = BOARD_COLS - 1 - state.pawn_pos[1]['x']
        net_inputs[p1_y][p1_x][2] = 1
        net_inputs[p0_y][p0_x][3] = 1

        if state.fence_num[0] == 8:
            for j in range(BOARD_ROWS):
                for i in range(BOARD_COLS):
                    net_inputs[j][i][5] = 1
        elif state.fence_num[0] > 0:
            for j in range(BOARD_ROWS):
                net_inputs[j][state.fence_num[0]-1][5] = 1

        if state.fence_num[1] == 8:
            for j in range(BOARD_ROWS):
                for i in range(BOARD_COLS):
                    net_inputs[j][i][4] = 1
        elif state.fence_num[1] > 0:
            for j in range(BOARD_ROWS):
                net_inputs[j][state.fence_num[1]-1][4] = 1

        for j in range(BOARD_ROWS-1):
            for i in range(BOARD_COLS-1):
                if state.hor_fence_pos[j, i] == 1:
                    net_inputs[BOARD_ROWS - 2 - j][BOARD_COLS - 2 - i][0] = 1   #障碍1的位置大小是6*6的,障碍2在1的基础上行或列+1
                    net_inputs[BOARD_ROWS - 2 - j][BOARD_COLS - 1 - i][0] = 1
                if state.ver_fence_pos[j, i] == 1:
                    net_inputs[BOARD_ROWS - 2 - j][BOARD_COLS - 2 - i][1] = 1
                    net_inputs[BOARD_ROWS - 1 - j][BOARD_COLS - 2 - i][1] = 1

    return net_inputs

def action_flip(action):
    if action < 12:
        act = 11 - action
    elif action >= 12 and action < 48:
        act = 59 - action    # a = 35-(a-12) +12 = 59 - a
    else:                       # a >= 48 and <84
        act = 131 - action   # a = 35-(a-48) +48 = 131 - a
    return act

def zero_play(judger, humanplayer=-1, print_state=False):       #for test
    alternator = judger.alternate()
    current_state = State()
    # last_last_state = State()
    # last_state = State()
    network = policy_value_network()
    tree = mcts_tree(network, judger.first_player)

    if judger.first_player == humanplayer:  #如果是人机对战，人先下的话就要手动再扩展一下根节点
        net_inputs = get_net_inputs(current_state, judger.first_player)
        net_inputs = np.expand_dims(net_inputs, 0)
        action_probs, value = tree.network.forward(net_inputs)
        action_probs = action_probs.flatten()

        legal_actions = judger.player[judger.first_player].get_actions(current_state)
        if judger.first_player == 1:
            for action in legal_actions:
                tree.rootNode.AddChild(action, judger.player[1 - judger.first_player].symbol, action_probs[action_flip(action)])
        else:
            for action in legal_actions:
                tree.rootNode.AddChild(action, judger.player[1 - judger.first_player].symbol, action_probs[action])
        tree.expanded.add(tree.rootNode)

    while True:

        player = next(alternator)
        if player.symbol != humanplayer:

            # if player == judger.player[0]: print('current player:p1')
            # elif player == judger.player[1]: print('current player:p2')



            ##################
            # test
            # current_state.pawn_pos[0]['y'] = 0
            # current_state.pawn_pos[1]['x'] = 5
            # net_inputs = get_net_inputs(current_state, player.symbol)

            # # _net_inputs = net_inputs.transpose((2,0,1))
            # print(net_inputs)


            # net_inputs = np.expand_dims(net_inputs, 0)
            # action_probs, value = network.forward(net_inputs)
            # action_probs = action_probs.flatten()
            
            # legal_actions = player.get_actions(current_state)
            # act_prob_dict = dict()
            # for action in legal_actions:
            #     if player.symbol == 1:
            #         act_prob_dict[action] = action_probs[action_flip(action)]
            #     else:
            #         act_prob_dict[action] = action_probs[action]

            # action = max(act_prob_dict.items(), key=lambda node: node[1])[0]

            # break
            
            ##################

            ##############
            # 网络直接走棋
            net_inputs = get_net_inputs(current_state, player.symbol)

            # _net_inputs = net_inputs.transpose((2,0,1))
            # print(_net_inputs)

            net_inputs = np.expand_dims(net_inputs, 0)
            action_probs, value = network.forward(net_inputs)
            action_probs = action_probs.flatten()
            
            legal_actions = player.get_actions(current_state)
            act_prob_dict = dict()
            for action in legal_actions:
                if player.symbol == 1:
                    act_prob_dict[action] = action_probs[action_flip(action)]
                else:
                    act_prob_dict[action] = action_probs[action]

            _action = sorted(act_prob_dict.items(), key=lambda node: node[1], reverse= True)
            action = max(act_prob_dict.items(), key=lambda node: node[1])[0]
            # action = np.random.choice(legal_actions)
            player.do_action(action, current_state)
            # break
            ############

            # action, _, value = tree.zero_mcts(current_state, judger, 1200)
            # player.do_action(action, current_state)
            # print('win_rate:', value)
            ############

            # legal_actions = player.get_actions(current_state)

            # if legal_actions == []:
            #     last_last_state.print_state()
            #     last_state.print_state()
            #     current_state.print_state()

            # action = np.random.choice(legal_actions)
            # last_last_state = last_state.Clone()
            # last_state = current_state.Clone()
            # player.do_action(action, current_state)
            ###################

            # action = np.random.choice(legal_actions)
            # player.do_action(action, current_state)
        else:
            legal_actions = player.get_actions(current_state)
            while True:
                action = int(input("Enter your action:"))
                if action not in legal_actions:
                    print('invalid action!')
                else:
                    break
            
            tree.expanded.discard(tree.rootNode)        #复用子树,人机对战时也要更新树
            for nod in tree.rootNode.childNodes:
                if nod.action == action:
                    win_rate = nod.Q
                    tree.rootNode = nod
                    break
            tree.rootNode.parentNode = None
            tree.rootNode.depth = 0

            player.do_action(action, current_state)

        if print_state:
            time.sleep(0.1)
            current_state.print_state()
            # print('win_rate:', value[0])
        
        if current_state.is_end():
            current_state.print_state()
            return current_state.winner

class quoridor_main:

    def __init__(self, judger, in_batch_size=512):    #128
        self.epochs = 5
        self.log_file = open(os.path.join(os.getcwd(), 'log_file.txt'), 'w')
        self.judger = judger
        self.batch_size = in_batch_size
        self.buffer_size = 10000
        self.global_step = 0
        self.kl_targ = 0.025
        self.learning_rate = 0.001    #5e-3    #    0.001
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.network = policy_value_network()
        self.state = None

    def selfplay(self):
        self.state = State()
        alternator = self.judger.alternate()
        tree = mcts_tree(network= self.network, first_player = self.judger.first_player)
        states, mcts_probs, current_players = [], [], []
        round_cnt = 0
        while True:

            player = next(alternator)

            action, act_probs, win_rate = tree.zero_mcts(self.state, self.judger, itermax = 500, selfplay= True)
            states.append(get_net_inputs(self.state, player.symbol))
            prob = np.zeros(ACTION_SPACE)
            for item in act_probs.items():
                if player.symbol == 0:
                    prob[item[0]]=item[1]
                elif player.symbol == 1:
                    prob[action_flip(item[0])]=item[1]
            mcts_probs.append(prob)
            current_players.append(player.symbol)

            player.do_action(action, self.state)
            round_cnt += 1
            # self.state.print_state()
            # print(self.state.dead_map)
            if round_cnt%16 == 0:
                print('selfplay round:', round_cnt, 'win_rate:', win_rate)
                self.state.print_state()

            if round_cnt > 120:
                winners_z = np.zeros(len(current_players))
                return zip(states, mcts_probs, winners_z), round_cnt

            if self.state.is_end():
                print('winner:', self.state.winner)
                winners_z = np.zeros(len(current_players))
                winners_z[np.array(current_players) == self.state.winner] = 1.0
                winners_z[np.array(current_players) != self.state.winner] = -1.0
                return zip(states, mcts_probs, winners_z), round_cnt

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        #print("training data_buffer len : ", len(self.data_buffer))
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        # print(np.array(winner_batch).shape)
        # print(winner_batch)
        winner_batch = np.expand_dims(winner_batch, 1)
        # print(winner_batch.shape)
        # print(winner_batch)
        start_time = time.time()
        old_probs, old_v = self.network.forward(state_batch)
        for i in range(self.epochs):
            accuracy, loss, self.global_step = self.network.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                             self.learning_rate * self.lr_multiplier)    #
            new_probs, new_v = self.network.forward(state_batch)
            kl_tmp = old_probs * (np.log((old_probs + 1e-10) / (new_probs + 1e-10)))
            # print("kl_tmp.shape", kl_tmp.shape)
            kl_lst = []
            for line in kl_tmp:
                # print("line.shape", line.shape)
                all_value = [x for x in line if str(x) != 'nan' and str(x)!= 'inf']#除去inf值
                kl_lst.append(np.sum(all_value))
            kl = np.mean(kl_lst)
            # kl = scipy.stats.entropy(old_probs, new_probs)
            # kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        self.network.save(self.global_step)
        print("train using time {} s".format(time.time() - start_time))

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},loss:{},accuracy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, accuracy, explained_var_old, explained_var_new))
        self.log_file.write("kl:{:.5f},lr_multiplier:{:.3f},loss:{},accuracy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, accuracy, explained_var_old, explained_var_new) + '\n')
        self.log_file.flush()

    def train(self):
        batch_iter = 0
        try:
            while(True):
                batch_iter += 1
                play_data, episode_len = self.selfplay()
                print("batch i:{}, episode_len:{}".format(batch_iter, episode_len))

                self.data_buffer.extend(play_data)

                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                        
                # print("batch i:{}, episode_len:{}".format(batch_iter, episode_len))

        except KeyboardInterrupt:
            self.log_file.close()
            self.network.save(self.global_step)

if __name__ == '__main__':
    player0 = Player(0)
    player1 = Player(1)
    judger = Judger(player0, player1, first_player = 0)
    # while True:
    zero_play(judger, humanplayer=-1, print_state= True)

    # quoridor = quoridor_main(judger)
    # quoridor.train()
    # while True:
    #     quoridor.selfplay()
