#!/usr/bin/env python3
from __future__ import annotations

import sys
import time

import numpy as np
from numba import cuda


class MazeProblem:
    """Represents data for instance of maze problem."""

    def __init__(self, width: int, height: int, step_reward: float, delay_reward, goal_reward: float, gamma: float,
                 epsilon: float, layout: list[list[str]]):
        self.width: int = width
        """Width of maze"""
        self.height: int = height
        """Height of maze"""
        self.step_reward: float = step_reward
        """Reward for taking a step onto an empty tile"""
        self.delay_reward: float = delay_reward
        """Reward for taking a step onto an delay tile"""
        self.goal_reward: float = goal_reward
        """Reward for taking a step onto an goal tile"""
        self.gamma: float = gamma
        """Discount factor for future rewards"""
        self.epsilon: float = epsilon
        """Stopping criteria for the value iteration algorithm, used to determine when the algorithm should terminate"""
        self.layout: list[list[str]] = layout
        """Stores structure of the maze. Used row major order. Value of layout[i][j] corresponds to i-th row
        and j-th position in row. Range of expected values should be {' ', '#', 'D', 'G'}."""

    def __str__(self):
        string = f"{self.width} {self.height} {self.step_reward} {self.delay_reward} {self.goal_reward} {self.gamma} " \
                 f"{self.epsilon}\n"
        for row in self.layout:
            string += f"{''.join(row)}\n"
        return string

    @staticmethod
    def load_from_file(file: str) -> MazeProblem:
        with open(file, 'r') as f:
            width, height, step_reward, delay_reward, goal_reward, gamma, epsilon = map(float, f.readline().split())
            lines = f.readlines()
        layout = [list(line.replace("\n", "")) for line in lines]
        return MazeProblem(int(width), int(height), step_reward, delay_reward, goal_reward, gamma, epsilon, layout)

@cuda.jit(device=True)
def next_state_is_valid_kernel(width, height, layout, state_i, state_j):
    i, j = state_i, state_j
    return 0 <= i < height and 0 <= j < width and layout[i, j] != -1

#    char_to_num = {' ': 0, 'G': 1, '#': -1, 'D': -2}
@cuda.jit(device=True)
def get_state_reward_kernel(state_tile, step_reward, delay_reward, goal_reward):
    if state_tile == 0:
        return step_reward
    elif state_tile == -2:
        return delay_reward
    elif state_tile == 1:
        return goal_reward
    return 0

@cuda.jit
def vi_kernel(width: int, height: int, step_reward: int, delay_reward: int, goal_reward: int, gamma: np.float64,
              layout, actions, subactions_probabilities, vi_values_old, vi_values_new, diff_matrix):
    i, j = cuda.grid(2)
    if j < width and i < height:
        state = layout[i, j]
        if state == -1:  # wall '#'
            vi_values_new[i, j] = 0
            return
        elif state == 1:  # goal 'G'
            vi_values_new[i, j] = 0
            return

        max_action_value = np.NINF
        for action in actions:
            action_value = 0
            for index, sub_action in enumerate(action):
                state_i = i + sub_action[0]
                state_j = j + sub_action[1]
                sub_action_probability = subactions_probabilities[index]
                if next_state_is_valid_kernel(width, height, layout, state_i, state_j):
                    reward = get_state_reward_kernel(layout[state_i, state_j], step_reward, delay_reward, goal_reward)
                    action_value += sub_action_probability * (reward + gamma * vi_values_old[state_i, state_j])
                else:
                    # If hitting a wall, stay in current state
                    reward = get_state_reward_kernel(layout[i, j], step_reward, delay_reward, goal_reward)
                    action_value += sub_action_probability * (reward + gamma * vi_values_old[i, j])
            if action_value > max_action_value:
                max_action_value = action_value
        vi_values_new[i, j] = max_action_value
    cuda.syncthreads()

    i, j = cuda.grid(2)
    if i < height and j < width:
        diff_matrix[i * width + j] = abs(vi_values_new[i, j] - vi_values_old[i, j])

    # cuda.syncthreads()
    # if i == 0 and j == 0:
    #     # max_diff[0] = np.NINF
    #     for row in range(height):
    #         for col in range(width):
    #             if diff_matrix[row * width + col] > max_diff[0]:
    #                 max_diff[0] = diff_matrix[row * width + col]

@cuda.reduce
def max_reduce(a, b):
    return max(a, b)


def main(instance_file, solution_file):
    instance = MazeProblem.load_from_file(instance_file)

    start_time = time.time()

    # todo implement VI algorithm
    vi_values_result = np.zeros((instance.height, instance.width))

    threads_per_block_x = 16
    threads_per_block_y = 16
    threads_per_block = (threads_per_block_x, threads_per_block_y)
    blocks_per_grid_x = (instance.width + threads_per_block_x - 1)
    blocks_per_grid_y = (instance.height + threads_per_block_y - 1)
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    char_to_num = {' ': 0, 'G': 1, '#': -1, 'D': -2}
    maze_layout = np.array([[char_to_num[char] for char in row] for row in instance.layout], dtype=np.int32)

    #  4 actions: left, right, top, down; 3 sub_actions: main, left, right
    actions = np.array([[[0, -1], [1, 0], [-1, 0]],
                        [[0, 1],[-1, 0], [1, 0]],
                        [[-1, 0], [0, -1], [0, 1]],
                        [[1, 0], [0, 1], [0, -1]]])

    sub_actions_probabilities = np.array([0.8, 0.1, 0.1], dtype=np.float64)  # main action, turn left, turn right

    vi_values_new = np.zeros((instance.height, instance.width), dtype=np.float64)
    vi_values_old = np.zeros((instance.height, instance.width), dtype=np.float64)

    d_actions = cuda.to_device(actions)
    d_sub_actions_probabilities = cuda.to_device(sub_actions_probabilities)
    d_layout = cuda.to_device(maze_layout)
    d_vi_values_new = cuda.to_device(vi_values_new)
    d_vi_values_old = cuda.to_device(vi_values_old)
    diff_matrix = np.zeros(instance.height * instance.width, dtype=np.float64)
    d_diff_matrix = cuda.to_device(diff_matrix)  # 1D matrix
    # max_diff = np.zeros(1, dtype=np.float64)
    # d_max_diff = cuda.to_device(max_diff)
    converged = False
    # diff_matrix = np.zeros(1, dtype=np.float64)
    while not converged:

        vi_kernel[blocks_per_grid, threads_per_block](instance.width, instance.height, instance.step_reward,
                                                      instance.delay_reward, instance.goal_reward, instance.gamma,
                                                      d_layout, d_actions, d_sub_actions_probabilities,
                                                      d_vi_values_old, d_vi_values_new, d_diff_matrix)
        # d_vi_values_new.copy_to_host(vi_values_new)

        # d_diff_matrix.copy_to_host(diff_matrix)
        # max_change = diff_matrix.max()
        max_change = max_reduce(d_diff_matrix)
        converged = max_change <= instance.epsilon

        d_vi_values_old, d_vi_values_new = d_vi_values_new, d_vi_values_old
        # vi_values_old = vi_values_new.copy()

        # d_max_diff.copy_to_host(max_diff)
        # print(max_diff)
        # converged = max_diff[0] <= instance.epsilon
        # d_vi_values_old, d_vi_values_new = d_vi_values_new, d_vi_values_old
    d_vi_values_new.copy_to_host(vi_values_new)
    np.copyto(vi_values_result, vi_values_new)

    elapsed_time = time.time() - start_time
    print(f"Elapsed VI time: {elapsed_time:.3f} seconds")

    with open(solution_file, 'w') as f:
        for row in vi_values_result:
            f.write(' '.join(map(str, row)) + '\n')


if __name__ == "__main__":
    instance_path = sys.argv[1]
    solution_path = sys.argv[2]
    main(instance_path, solution_path)
