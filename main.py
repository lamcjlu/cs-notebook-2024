import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from abc import ABC, abstractmethod
from collections import deque
from pettingzoo.classic import tictactoe_v3
from typing import List

from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
from functools import reduce
import operator
import seaborn as sns
import pandas as pd
import pickle

from torch import autocast
from torch.distributions import Categorical
from torchrl.envs.libs import pettingzoo

torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_device('cuda')


class Environment:
    def __init__(self, size=(21, 21, 21), diff=1, risk_prob=0.1, death_prob=0.01):
        self.size = size  # Grid size, e.g., (100, 100, 100) for 3D
        self.sz_tensor = torch.tensor(size)
        self.vol = size[0] * size[1] * size[2]
        self.diff = diff  # A difficulty or variability factor
        self.dim = len(size)
        self.target = (0, 0, 0, 0)  # energy-p, wall-p, risk-p, death-p by volume or by cells, index by 0
        self.isRiskCluster = True

        # main game -> min (time(steps) * total_energy) -> compared with (l2_distance * harmonic_or_mean_or_70_or_max_energy). if agent_score < estimated_score, good. if a_s > e_s, improve. hm7m is param tightened by agent performance
        # minimize energy of path -> main objective
        self.energy = torch.clamp(torch.normal(0.2, 0.5, size=self.size), min=0, max=3)
        # cannot transpose to wall, if slip into wall -> terminate , death ,-r
        self.wall = torch.zeros(size)
        # risk will lead to slip to a random neighbor cell
        self.risk = torch.mul(torch.rand(size), risk_prob)
        # if death will terminal state, negative reward (-r)
        self.death = torch.mul(self.risk, death_prob)

        self.init_properties()
        # map = not wall(death + energy * risk)) // Energy has been normalised to 0-1
        self.map = torch.add(
            torch.mul(
                (self.energy.clone() * (1 + self.risk.clone()) / self.energy.max()),
                torch.logical_not(self.wall.clone()).half()
            ),
            self.death.clone())

    def set_diff(self):
        print(self.vol)
        if self.diff == 1:
            self.target = (1, self.vol * 0.1, 0.1, 0.01)
            print("Setting Diff: ", self.diff, " -> ", self.target)
            return

        if self.diff == 2:
            self.target = (1.2, self.vol * 0.2, 0.2, 0.05)
            print("Setting Diff: ", self.diff, " -> ", self.target)
            return

        if self.diff == 3:
            self.target = (2, self.vol * 0.3, 0.3, 0.1)
            print("Setting Diff: ", self.diff, " -> ", self.target)
            return

    def init_properties(self):
        # Initialize grids with normal distributions
        self.diff = np.random.randint(1, 4)
        self.set_diff()
        volume = reduce(operator.mul, self.size, 1)
        print("INIT: Volume", volume)
        print("INIT: target: ", self.target)
        mask_r = int(max(self.size) * 0.1)  # 70% limit of any dim

        def create_hot_tensor(shape, min_value, max_value):
            dims = len(shape)
            ranges = [torch.arange(s, dtype=torch.float16) - (s - 1) / 2.0 for s in shape]
            grid = torch.meshgrid(ranges, indexing='ij')
            dist_matrix = torch.stack([torch.abs(grid[dim]) for dim in range(dims)]).max(0).values
            steps = max((s - 1) / 2 for s in shape)
            increment = (max_value - min_value) / steps if steps != 0 else 0
            values = min_value + (steps - dist_matrix) * increment
            values = torch.clamp(values, min=min_value, max=max_value).int()  # Ensure values are within specified range
            return values

        def set_random_dim_to_one(size):
            return tuple(np.where(np.arange(len(size)) == np.random.randint(len(size)), 2, size))

        def apply_hot_region(tensor, center, size, min_value, max_value, mode):
            if mode == 0:
                hot_tensor = create_hot_tensor(size, min_value, max_value)
            elif mode == 1:
                size = set_random_dim_to_one(size)  # shrink size by x dim
                hot_tensor = torch.ones(size)

            # Calculate slice for each dimension for the parent tensor
            parent_slices = []
            hot_tensor_slices = []

            for c, s, es in zip(center, size, tensor.shape):
                # Start and end points for the slice on the parent tensor
                start = max(0, c - s // 2)
                end = min(c + (s + 1) // 2, es)

                # Corresponding start and end points on the hot_tensor
                hot_start = max(0, s // 2 - c if c < s // 2 else 0)
                hot_end = s - max(0, (c + (s + 1) // 2) - es)

                # Append the slices to the lists
                parent_slices.append(slice(start, end))
                hot_tensor_slices.append(slice(hot_start, hot_end))

            # Convert lists to tuples
            parent_slices = tuple(parent_slices)
            hot_tensor_slices = tuple(hot_tensor_slices)

            # Place the correctly sliced hot_tensor into the parent tensor
            tensor[parent_slices] = hot_tensor[hot_tensor_slices]
            return tensor

        def set_tensor_edges_to_one(tensor):
            """
            Sets the edges of an n-dimensional tensor to 1.

            Args:
            - tensor (torch.Tensor): An n-dimensional tensor.

            Returns:
            - torch.Tensor: The modified tensor with its edges set to 1.
            """
            # Iterate over each dimension and set the edge indices to 1
            for dim in range(tensor.ndim):
                # Get a list of all slice(None) initially which means select everything along each dimension
                indexer = [slice(None)] * tensor.ndim

                # Set the first and last index of the current dimension to 1
                indexer[dim] = 0
                tensor[tuple(indexer)] = 1
                indexer[dim] = -1
                tensor[tuple(indexer)] = 1

            return tensor

        def rand_center(tensor, r):
            # Ensure that the generated center is at least `r` away from the edges of the tensor
            return tuple(np.random.randint(low=r, high=dim - r) if dim > 2 * r else r for dim in tensor.shape)

        def tensor_vol(dim, vol):
            k = 1
            random_matrix = torch.clamp(torch.rand(dim), max=k)
            return tuple(torch.round(random_matrix * vol).int().tolist())

        print("INIT: Energy", self.energy.mean() > self.target[0])
        energy_flag = True
        i = 0
        while energy_flag:
            self.energy = apply_hot_region(
                self.energy,
                rand_center(self.energy, mask_r),
                tensor_vol(self.dim, self.vol * 0.003),
                0,
                3,
                mode=0
            )
            i += 1
            if (self.energy.mean() > self.target[0]) or (i == 50000):
                energy_flag = False
                print("if ", self.energy.mean() > self.target[0], " or ", (i == 50000), "flag: ", energy_flag)
                print("self.energy.mean: ", self.energy.mean())

        print("INIT: Wall", self.wall.sum() > (self.target[1]))
        wall_flag = True
        i = 0
        while wall_flag:
            self.wall = apply_hot_region(
                self.wall,
                rand_center(self.wall, mask_r),
                tensor_vol(self.dim, self.vol * 0.001),
                1,
                1,
                mode=1
            )
            i += 1
            if (self.wall.sum() > self.target[1]) or (i == 10000):
                wall_flag = False
                print("if ", self.wall.sum() > self.target[1], " or ", (i == 10000), "-> flag: ", wall_flag)
                self.wall = set_tensor_edges_to_one(self.wall.clone())
                self.wall = torch.bernoulli(self.wall)

        print("INIT: Risk", self.risk.mean() > (self.target[2]))
        risk_flag = True
        i = 0
        if self.isRiskCluster:
            while risk_flag:
                self.risk = apply_hot_region(
                    self.risk,
                    rand_center(self.risk, mask_r),
                    tensor_vol(self.dim, self.vol * 0.001),
                    1,
                    10,
                    mode=0
                )
                i += 1
                if (self.risk.mean() / 10 > self.target[2]) or (i == 10000):
                    risk_flag = False
                    print("if ", self.risk.mean() / 10 > self.target[2], " or ", (i == 10000), "flag: ", risk_flag)
                    self.risk = torch.mul(self.risk, 0.1)

        print("INIT: Death", self.risk.mean() * self.target[3])
        self.death = torch.bernoulli(torch.mul(self.risk, self.target[3]))

    def env_global_features(self, len=3):
        # Function to calculate feature sums for all axes
        def compress_nd_matrix_sum(data) -> torch.tensor:
            """
            Compress an n-dimensional matrix by summing along each axis using PyTorch.

            Parameters:
                data (torch.Tensor): An n-dimensional PyTorch tensor.
            """
            n_dimensions = data.ndim
            compressed_results = []
            for axis in range(n_dimensions):
                # Sum along the current axis and store the result
                axis_sum = torch.sum(data, dim=axis)
                compressed_results.append(axis_sum)
            return torch.stack(compressed_results)

        if len == 1:
            return torch.stack([compress_nd_matrix_sum(self.map.clone())
                                ])
        elif len == 2:
            return torch.stack([compress_nd_matrix_sum(self.map.clone()),
                                compress_nd_matrix_sum(self.wall.clone())
                                ])
        elif len == 3:
            return torch.stack([compress_nd_matrix_sum(self.energy.clone()),
                                compress_nd_matrix_sum(self.risk.clone()),
                                compress_nd_matrix_sum(self.wall.clone()),
                                ])
        elif len == 4:
            return torch.stack([compress_nd_matrix_sum(self.energy.clone()),
                                compress_nd_matrix_sum(self.risk.clone()),
                                compress_nd_matrix_sum(self.wall.clone()),
                                compress_nd_matrix_sum(self.death.clone())
                                ])
        elif len == 5:
            return torch.stack([compress_nd_matrix_sum(self.energy.clone()),
                                compress_nd_matrix_sum(self.risk.clone()),
                                compress_nd_matrix_sum(self.wall.clone()),
                                compress_nd_matrix_sum(self.death.clone()),
                                compress_nd_matrix_sum(self.map.clone())
                                ])

    def env_local_state(self, loc, len=3):
        def return_stack(loc, len=3):
            if len == 1:
                return torch.stack([pad_local_state(self.map.clone(), loc, len)
                                    ])
            elif len == 2:
                return torch.stack([pad_local_state(self.map.clone(), loc, len),
                                    pad_local_state(self.wall.clone(), loc, len)
                                    ])
            elif len == 3:
                return torch.stack([pad_local_state(self.energy.clone(), loc, len),
                                    pad_local_state(self.risk.clone(), loc, len),
                                    pad_local_state(self.wall.clone(), loc, len)
                                    ])
            elif len == 4:
                return torch.stack([pad_local_state(self.energy.clone(), loc, len),
                                    pad_local_state(self.risk.clone(), loc, len),
                                    pad_local_state(self.wall.clone(), loc, len),
                                    pad_local_state(self.death.clone(), loc, len)
                                    ])
            elif len == 5:
                return torch.stack([pad_local_state(self.energy.clone(), loc, len),
                                    pad_local_state(self.risk.clone(), loc, len),
                                    pad_local_state(self.wall.clone(), loc, len),
                                    pad_local_state(self.death.clone(), loc, len),
                                    pad_local_state(self.map.clone(), loc, len)
                                    ])

        def pad_local_state(input_tensor, location, distance):
            # Create slice objects for each dimension
            dim = input_tensor.dim()
            slices = []
            for i in range(dim):
                start = location[i] - distance
                end = location[i] + distance + 1  # +1 for exclusive end
                slices.append(slice(max(0, start), min(input_tensor.size(i), end)))

            # Extract the slice from the tensor
            extracted = input_tensor[slices]

            # Determine the shape of the full result with padding
            full_shape = [2 * distance + 1] * dim

            # Create a tensor of zeros with the target shape
            result = torch.zeros(full_shape)

            # Calculate the slices for inserting the extracted data into the result
            insert_slices = [slice(max(0, distance - (location[i] - slices[i].start)),
                                   max(0, distance - (location[i] - slices[i].start)) + extracted.size(i)) for i in
                             range(dim)]

            # Place the extracted slice into the padded result tensor
            result[insert_slices] = extracted

            return result

        return return_stack(loc, len)

    def roll(self, location):
        r = torch.rand(1)
        print("ENV.roll(): ", r, self.index_tensor(self.risk, location))

        if self.index_tensor(self.risk, location) > r:
            return True, self.risk[location]
        else:
            return False, self.risk[location]

    def get_cell_depreciated(self, location):
        if not torch.is_tensor(location):
            location = torch.tensor(location)

        print("ENV.get_cell: ", location)
        clamped_locations = torch.min(location, torch.tensor(self.map.shape) - 1)
        return (self.energy[clamped_locations],
                self.risk[clamped_locations],
                self.wall[clamped_locations],
                self.death[clamped_locations],
                self.map[clamped_locations])

    def index_tensor(self, target_tensor, locations):
        locations = torch.as_tensor(locations)

        # Ensure locations is a 2D tensor
        if locations.dim() == 1:
            locations = locations.unsqueeze(0)

        # Clamp locations for each dimension
        clamped_locations = torch.stack([
            torch.clamp(locations[:, dim], 0, size - 1)
            for dim, size in enumerate(target_tensor.shape[:locations.size(1)])
        ], dim=1)

        # Use advanced indexing to select elements
        return target_tensor[tuple(clamped_locations.t())]

    def get_cell(self, location):

        # if not torch.is_tensor(location):
        #    location = torch.tensor(location, dtype=torch.long)
        location = torch.as_tensor(location)
        return (self.index_tensor(self.energy, location),
                self.index_tensor(self.risk, location),
                self.index_tensor(self.wall, location),
                self.index_tensor(self.death, location),
                self.index_tensor(self.map, location))


class AgentWrapper:
    def __init__(self, agent, info):
        self.agent = agent
        self.team = info['team']
        self.start_pos = info['current_pos']
        self.end_pos = info['end_pos']

        # Track agent's state, actions, and rewards
        self.states = []  # Initial state
        self.location = []
        self.actions = []
        self.rewards = []
        self.violations = []
        self.total_reward = 0

    def choose_action(self, state):
        action = self.agent.choose_action(state)
        self.actions.append(action)
        return action

    def receive_feedback(self, reward, next_state, done):
        self.rewards.append(reward)
        self.states.append(next_state)
        if done:
            self.episode_end()

    def episode_end(self):
        # Aggregate rewards and update the model at the end of an episode
        print(f"AgentWrapper.episode_end: {self.team}")
        self.total_reward = sum(self.rewards)
        for i in range(len(self.rewards)):
            state = self.states[i]
            action = self.actions[i]
            reward = self.rewards[i]
            next_state = self.states[i + 1] if i + 1 < len(self.states) else None
            done = i == len(self.rewards) - 1
            self.agent.update_model(state, action, reward, next_state, done)

        self.rewards.clear()
        self.actions.clear()
        self.states.clear()

    def review_performance(self):
        # Method to review performance metrics of the agent
        print(f"Total reward accumulated in last episode: {self.total_reward}")


class AbstractRLAgent(ABC):
    def __init__(self, state_size, action_size, info):
        self.current_pos = info['current_pos']
        self.end_pos = info['end_pos']
        self.team = info['team']

        self.action_size = action_size
        self.state_size = state_size

        self.cumulative_reward = 0
        self.history = None
        self.state = None
        self.action = None

    @abstractmethod
    def choose_action(self, state):
        """
        Get the action to take based on the current state.

        Args:
            state: The current state in the environment.

        Returns:
            An integer representing the chosen action.
        """
        return self.policy.decide(state)

    @abstractmethod
    def update_model(self, state, action, reward, next_state, done):
        """
        Update the model based on the transition.

        Args:
            state: The current state from which the action was taken.
            action: The action taken.
            reward: The reward received after taking the action.
            next_state: The state transitioned to after the action.
            done: Boolean indicating if the episode has terminated.
        """
        # Example learning procedure based on the reward and next state
        pass


class HelloAgent(AbstractRLAgent):
    class LocalNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.CELU(),
                nn.Linear(64, 32),
                nn.CELU(),
                nn.Linear(32, output_dim),
                nn.Softmax(dim=-1)
            )
            self.filename = f"models/NN-HelloAgent-{input_dim}-{output_dim}.pt"

        def forward(self, x):
            return self.layers(x)

        def save_model(self):
            # Saving as TorchScript
            model_scripted = torch.jit.script(self)
            model_scripted.save(self.filename)
            print(f"Model saved {self.filename}")

        def load_model(self):
            model_loaded = torch.jit.load(self.filename)
            print(f"Model loaded from {self.filename}")
            return model_loaded

    # @autocast(device_type=torch.get_default_device())
    def __init__(self, state_size, action_size, info):
        super().__init__(state_size, action_size, info)  # Ensuring base class initialization
        self.model = self.LocalNN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=info['lr'])
        self.criterion = nn.MSELoss()

        self.epsilon = 0.1
        self.gamma = 0.99

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            isrand = True
            action = np.random.randint(0, self.action_size)
        else:
            isrand = False
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float)
                q_values = self.model(state)
                action = torch.argmax(q_values).item()
        print(f"HelloAgent.choose_action: {action}, RNG:{isrand}")
        return action

    def update_model(self, state, action, reward, next_state, done):
        print(f"HelloAgent.update_model: {state}, {action}, {reward}, {next_state}, {not done}")
        if not done:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            q_values = self.model(state)
            next_q_values = self.model(next_state)

            max_next_q_value = torch.max(next_q_values).item()
            target_q_value = reward + self.gamma * max_next_q_value

            loss = self.criterion(q_values[action], target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class NN(nn.Module):
    # https://r-knott.surrey.ac.uk/Fibonacci/fibtable.html
    def __init__(self, input_dim, output_dim, mode=0):
        super(NN, self).__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.mode == 0:
            print("NN: ", self.mode)
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 610),
                nn.SELU(),
                nn.Linear(610, 377),
                nn.SELU(),
                nn.Linear(377, 144),
                nn.SELU(),
                nn.Linear(144, 233),
                nn.SELU(),
                nn.Linear(233, output_dim),
                nn.Softmax(dim=-1)
            )
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # Assuming a learning rate is defined

        elif self.mode == 1:
            # Dummy behavior for random output, could be designed better based on specific requirements
            self.layers = None

    def forward(self, x):
        if self.mode == 0:
            return self.layers(x)
        elif self.mode == 1:
            return torch.rand((x.shape[0], self.output_dim))  # Random tensor with the same batch size as input

    def save_model(self):
        filename = f"Zeo-NN-{self.mode}-{self.input_dim}-{self.output_dim}.pt"
        # Saving as TorchScript
        model_scripted = torch.jit.script(self)
        model_scripted.save(filename)
        print(f"Model saved as {filename}")

    def load_model(self, filepath):
        model_loaded = torch.jit.load(filepath)
        print(f"Model loaded from {filepath}")
        return model_loaded


class Observer:
    def __init__(self):
        self.env = Environment()

        self.agent_step_limit = self.env.vol * self.env.vol
        self.MARL = True
        self.LoadMap = False
        self.LoadNavModel = False
        self.LoadPvPModel = False
        self.ResumeNav = False
        self.ResumePvP = False

        self.agent_store = []
        self.log = []
        self.hard_L = self.env.map.max()

        self.transitions = {
            0: torch.tensor([2, 0, 0]),
            1: torch.tensor([-2, 0, 0]),
            2: torch.tensor([0, 2, 0]),
            3: torch.tensor([0, -2, 0]),
            4: torch.tensor([0, 0, 2]),
            5: torch.tensor([0, 0, -2]),
            6: torch.tensor([1, 1, 0]),
            7: torch.tensor([1, -1, 0]),
            8: torch.tensor([1, 0, 1]),
            9: torch.tensor([1, 0, -1]),
            10: torch.tensor([-1, 1, 0]),
            11: torch.tensor([-1, -1, 0]),
            12: torch.tensor([-1, 0, 1]),
            13: torch.tensor([-1, 0, -1]),
            14: torch.tensor([0, 0, 0])
            # 15: None # TeleportAgent(exits, distance) - handled by a pointer network, or arm -> prevent slip
        }
        self.neighbours = torch.tensor([
            [2, 0, 0],
            [-2, 0, 0],
            [0, 2, 0],
            [0, -2, 0],
            [0, 0, 2],
            [0, 0, -2],
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, 1],
            [1, 0, -1],
            [-1, 1, 0],
            [-1, -1, 0],
            [-1, 0, 1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, -1],
            [0, 0, 0]
        ])
        self.spawn_agents()
        # self.current_position = torch.tensor([5, 5, 5])

    def train(self):
        # Initialization of a training session
        training_active = True
        iteration = 0
        # init agents

        while training_active:
            iteration += 1
            all_agents_done = True

            for agent_wrapper in self.agent_store:
                # Generate the current state for the agent based on its last observed location
                state = self.observation(
                    agent_wrapper.location[-1],
                    agent_wrapper.end_pos,
                    agent_wrapper.team
                )

                # Agent[i] chooses an action based on the current state
                action = agent_wrapper.choose_action(state)

                # Environment processes the action and returns the reward and next observation
                reward, next_state, terminate = self.step(agent_wrapper, action)

                # Update the agent model based on the action's outcomes
                agent_wrapper.receive_feedback(reward, next_state, terminate)
                if not terminate:
                    all_agents_done = False

            if all_agents_done:
                print(f"Observer.train(): Training ending at iteration {iteration}")
                for agent_wrapper in self.agent_store:
                    agent_wrapper.episode_end()

                if iteration == 1000:
                    training_active = False

    def step(self, agent, action, safety=True, is_slip_activated=True):
        """ agent -> AgentWrapper, action -> int """
        penalty = 0

        next_position = self.transitions[action] + torch.as_tensor(agent.location[-1])
        valid = self.valid_env_move(newpos=next_position, agent=agent)

        if not valid:
            agent.violations.append(self.env.get_cell(next_position))
            print("OBS.step: Invalid Move")
            terminate = True
            return (self.reward(agent=agent, loc=next_position, cell=self.env.get_cell(next_position), invalid=True),
                    self.observation(next_position, agent.end_pos, agent.team),
                    terminate)

        # apply slip, if true slides to random neighbour even if unsafe
        if is_slip_activated:
            roll, risk = self.env.roll(next_position)
            if roll:
                old_pos = next_position
                a = np.random.randint(0, len(self.transitions))
                next_position = self.transitions[a] + torch.as_tensor(agent.location[-1])
                valid = self.valid_env_move(newpos=next_position, agent=agent)

                if not valid:
                    print("OBS.step.slipped: Invalid Move")
                    terminate = True
                    return (
                        self.reward(agent=agent, loc=next_position, cell=self.env.get_cell(next_position),
                                    invalid=True),
                        self.observation(next_position, agent.end_pos, agent.team),
                        terminate)

        # check hard safety constraints
        if safety and self.l(mode='hard', locations=next_position, intensity=0.99):
            agent.violations.append(self.env.get_cell(next_position))
            print("OBS.step: Hard Constraint Violation, seeding next valid move")
            terminate = False
            all_indices = list(range(len(self.transitions)))  # Create a list of all possible indices
            np.random.shuffle(all_indices)  # Shuffle to ensure randomness

            for index in all_indices:  # Iterate over each index only once
                next_position = self.transitions[index] + agent.location[-1]
                valid = self.valid_env_move(newpos=next_position, agent=agent)
                if valid and self.l(mode='hard', locations=next_position, intensity=0.99):
                    print(f"Valid move found at index {index}, position {next_position}")
                    break

        # Calculate the reward based on the properties of the cell
        reward = self.reward(agent, next_position,
                             self.env.get_cell(next_position),
                             invalid=terminate,
                             slip_roll=roll)

        # check hard safety constraints

        # Return the reward and the next observation
        print("OBS.step.reward: ", reward)
        next_observation = self.observation(next_position, agent.end_pos, agent.team)
        return reward, next_observation, terminate

    def reward_old(self, agent, loc, cell, invalid=False, slip_roll=False):
        energy, risk, wall, death, m = cell

        def objective(agent, loc, m):
            # Objective function to be minimised
            # agent -> AgentWrapper, loc -> torch.tensor, m -> torch.tensor
            # Returns a scalar value representing the objective function
            return torch.sum(torch.abs(loc - agent.end_pos))

        penalty_invalid = -1
        penalty_slip = -0.5
        penalty_death = -1

        if slip_roll:
            return penalty_slip

        if invalid:
            return self.l('soft', loc, intensity=0.99)

        return m + self.l('soft', loc, intensity=0.99)

    def reward(self, agent, loc, cell, invalid=False, slip_roll=False, death=False):
        energy, risk, wall, death_state, m = cell
        penalty = -m + self.l('soft', loc, intensity=0.99)  # default penalty calculation

        # Apply the penalty adjustments based on conditions
        if invalid:
            penalty *= 3
        if slip_roll:
            penalty *= 1.6
        if death or death_state:
            penalty *= 4

        # Check if the agent reached the end position
        if torch.equal(agent.end_pos, loc):
            # Reward the agent based on reaching the target and the safety of the path
            penalty = -5 * penalty  # Positive reward, 5 times the negated penalty
            if slip_roll:
                penalty *= 0.4

            # Calculate distance to end position and apply a logarithmic scale discount
            distance_to_end = torch.norm(agent.end_pos - loc)
            max_distance = 2 * torch.prod(self.env.sz_tensor).item()
            discount_factor = torch.log1p(distance_to_end / max_distance)
            discount = torch.clamp(discount_factor, max=0.5)
            penalty *= (1 - discount)  # Apply discount to the reward

        return penalty

    def valid_env_move(self, newpos, agent):
        # Check if the agent can move in the specified direction
        energy, risk, wall, death, map = self.env.get_cell(newpos)
        if wall.max() == 1:
            print("OBS.valid_env_move: False")
            return False

        if newpos.min() >= 0 and newpos.max() < self.env.sz_tensor.min():
            print("OBS.valid_env_move: True")
            return True

    def CLF(self, locations):
        # should assess next worst possible state.
        # Theorm -> agent in a ok state should not die because of a nudge
        # like balancing a bottle, bottle at stable state ss should not fall down when k force is applied leading it to state sa or sb,
        # however sc may be dangerous and sf is prohibited.
        energy, risk, wall, death, map = self.env.get_cell(locations)
        return

    def l(self, mode, locations, intensity=0.99):
        q = torch.tensor([0.38, 0.5, 0.62])
        energy, risk, wall, death, map = self.env.get_cell(locations + self.neighbours)
        # print("OBS.l: ", energy.shape, risk.shape, wall.shape, death.shape, map.shape)
        if mode == 'soft':  # reward shaping and considers any next state
            # energy converges to 0, as min risk = 0
            # you are now in a "high" region that is not perfered
            score = (1 + risk.max()) * (1 + energy)
            score = torch.quantile(score.flatten().float(), q.float())
            print("OBS.l.soft.tq.score: ", score)
            score = torch.sum(score)
            print("OBS.l.soft.score: ", score)
            return -score

        elif mode == 'hard':  # hard safety rule
            # Checks next state if death as a hard limit.
            # Does not allow agent to move to any cell adjacent to death
            print("OBS.l.hard.bool: d, H*L ||", death.max(), self.hard_L * intensity)
            if death.max():
                print("OBS.l.hard.death: True")
                return True

            if self.hard_L * intensity >= map.max():
                print("OBS.l.hard.cell_is_too_hard: True")
                return True

            else:
                print("OBS.l.hard: False")
                return False

        # ensure the trajectory converges to stable state,
        # eg ground state -> energy equlibruim -> some minimizeable state
        # policy: death=high, risk=high, wall=high
        # can be returned as something to be minimised in the obj
        # or operated as a constraint to clamp next possible states
        # or be used to evaluate all next possible states

        # intensity defines max acceptable state by evaluating worse case
        # can be clamped by the mean or upper quantile l of l2 path.
        # mode decides soft, hard, or expect
        # soft is used for reward shaping, hard is used for constraint, expect is used for evaluation i states
        # possible to dampen next state based on previous i states
        # by exposing self to less risk energy state
        # target can be lower than threshold or dynamic

    def plot_space(self, mode):
        if mode == 0:
            print("env.energy: ", self.env.energy.min(), self.env.energy.mean(), self.env.energy.max())
            ipv.figure()
            ipv.volshow(self.env.energy, level=[0.1, 0.5, 0, 1, 3], opacity=0.03, level_width=0.1,
                        data_min=self.env.energy.min(), data_max=self.env.energy.max())
            ipv.show()

        if mode == 1:
            print("env.wall: ", self.env.wall.sum())
            ipv.figure()
            ipv.volshow(self.env.wall, level=[0, 1], opacity=0.05, level_width=0.1, data_min=self.env.wall.min(),
                        data_max=self.env.wall.max())
            ipv.show()

        if mode == 2:
            print("env.risk: ", self.env.risk.min(), self.env.risk.mean(), self.env.risk.max())
            ipv.figure()
            ipv.volshow(self.env.risk, level=[0, 0.25, 0.5, 0.75, 1], opacity=0.05, level_width=0.1,
                        data_min=self.env.risk.min(), data_max=self.env.risk.max())
            ipv.show()

        if mode == 3:
            print("env.death: ", self.env.death.min(), self.env.death.mean(), self.env.death.max())
            ipv.figure()
            ipv.volshow(self.env.death, opacity=0.05, level_width=0.1, data_min=self.env.death.min(),
                        data_max=self.env.death.max())
            ipv.show()

        if mode == 4:
            print("env.map: ", self.env.map.min(), self.env.map.mean(), self.env.map.max())
            ipv.figure()
            ipv.volshow(self.env.map, opacity=0.05, level_width=0.1, data_min=self.env.map.min(),
                        data_max=self.env.map.max())
            ipv.show()

    def plot_slice(self, mode, loc=[14, 14, 7], r=7):
        if mode == 0:
            print("env.energy.slice: ")
            ipv.figure()
            ipv.volshow(self.env.energy[loc[0] - r:loc[0] + r, loc[1] - r:loc[1] + r, loc[2] - r:loc[2] + r],
                        level=[0.1, 0.5, 0, 1, 3], opacity=0.03, level_width=0.1, data_min=0, data_max=3)
            ipv.show()

        if mode == 1:
            print("env.wall.slice: ")
            ipv.figure()
            ipv.volshow(self.env.wall[loc[0] - r:loc[0] + r, loc[1] - r:loc[1] + r, loc[2] - r:loc[2] + r],
                        level=[0, 1], opacity=0.05, level_width=0.1, data_min=0, data_max=1)
            ipv.show()

        if mode == 2:
            print("env.risk.slice: ")
            ipv.figure()
            ipv.volshow(self.env.risk[loc[0] - r:loc[0] + r, loc[1] - r:loc[1] + r, loc[2] - r:loc[2] + r],
                        level=[0, 0.25, 0.5, 0.75, 1], opacity=0.05, level_width=0.1, data_min=0, data_max=1)
            ipv.show()

    def plot_features(self, len=3):
        # Obtain data from your environment function
        tensors = self.env.env_global_features(len)
        print("OBS.plot_features.shape is: ", tensors.shape)
        # Determine the number of properties and scenarios dynamically
        num_properties, num_scenarios, _, _ = tensors.shape  # Assume shape is [num_properties, num_scenarios, 10, 10]

        # Prepare property names dynamically (if they are not predefined)
        property_names = [f'Property {i + 1}' for i in range(num_properties)]

        # Set up the figure based on the number of properties and scenarios
        fig, axs = plt.subplots(num_properties, num_scenarios, figsize=(num_scenarios * 6, num_properties * 6))

        # Check if we have a single row or column to adjust indexing
        if num_properties == 1 or num_scenarios == 1:
            axs = axs.reshape(num_properties, num_scenarios)

        # Loop through each property and scenario
        for i in range(num_properties):
            for j in range(num_scenarios):
                ax = axs[i, j]
                sns.heatmap(tensors[i, j], cmap='hot', ax=ax, cbar_kws={'label': f'Level for {property_names[i]}'})
                ax.set_title(f'{property_names[i]} - Scenario {j + 1}')

        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.show()  # Show the plot

    def observation(self, location, target, team, obs=3, k=3):
        # Method to build obs for NAVagent
        print("OBS: L, T = ", location, target)

        def obs_map(loc, obs, k):
            # 21 * 21 obs 441 pooled by k=3 to 49
            mean_egf = F.avg_pool2d(self.env.env_global_features(len=obs), kernel_size=k, stride=k)
            # map: map,wall: energy,risk,wall: || 1->147, 2->294, 3->441
            # energy,risk,wall,death: energy,risk,wall,death,map || 4->588, 5->735
            # build local exact highres observation with vision 5
            tensor_map = self.env.env_local_state(location, len=obs)
            # returns the global foggy look and the local true observation
            return torch.cat([mean_egf.flatten(), tensor_map.flatten()])

        def obs_nav(location, target):
            if not torch.is_tensor(location):
                location = torch.tensor(location)

                # Check if 'target' is a tensor, convert if not
            if not torch.is_tensor(target):
                target = torch.tensor(target)

            # builds the relative properties for the NN, eg abs distance from target
            return torch.cat([torch.tensor(location == target).flatten(),
                              torch.tensor(target - location).flatten()])

        return torch.cat([obs_map(location, obs, k), obs_nav(torch.tensor(location), target)])

    def spawn_agents_old(self):
        # state_space = sample reading from the environment
        def randpos():
            return torch.tensor([np.random.randint(0, self.env.map.size - 1) for _ in range(self.env.dim)])

        state_space = self.observation(location=torch.tensor([2, 2, 2]),
                                       target=torch.tensor([19, 19, 19]),
                                       team=0).shape[0]

        cfg_helloagent = {
            'lr': 1e-3,
            'criterion': nn.MSELoss,
            'epsilon': 0.1,
            'current_pos': randpos(),
            'end_pos': randpos(),
            'team': len(self.agent_store),
            'state': self.observation()
        }

        self.agent_store.append(
            AgentWrapper(HelloAgent(state_size=state_space,
                                    action_size=len(self.transitions),
                                    info=cfg_helloagent
                                    ), {'team': 0, 'start': torch.tensor([2, 2, 2]), 'end': [19, 19, 19]}))

        print("Observer.spawn_agents(): AgentStore | ", len(self.agent_store))

    def spawn_agents(self):
        def rand_2pos():
            return torch.tensor([np.random.randint(0, self.env.size[0] - 1) for _ in self.env.map.size()]), torch.tensor([np.random.randint(0, self.env.size[0] - 1) for _ in self.env.map.size()])

        # Randomly generate start and end positions
        spawn_pos, end_pos = rand_2pos()
        cfg_helloagent = {
            'lr': 1e-3,
            'criterion': nn.MSELoss,
            'epsilon': 0.1,
            'current_pos': spawn_pos,
            'end_pos': end_pos,
            'team': len(self.agent_store),  # Incremental team assignment
            'state': self.observation(spawn_pos, end_pos, len(self.agent_store))
        }

        # Append agent with the first state observation
        state_space = cfg_helloagent['state'].shape[0]
        agent = HelloAgent(state_size=state_space,
                           action_size=len(self.transitions),
                           info=cfg_helloagent)
        agent_wrapper = AgentWrapper(agent, cfg_helloagent)

        # Initialize with first state
        agent_wrapper.states.append(cfg_helloagent['state'])
        agent_wrapper.location.append(spawn_pos)

        self.agent_store.append(agent_wrapper)
        print("Observer.spawn_agents(): AgentStore | ", len(self.agent_store))

    def review(self):
        # load torchscripts for agents

        # snapshot environment
        return

    def review_PA(self):
        # check policy agent pairs and compare with different PA pairs
        return


if __name__ == "__main__":
    obs = Observer()
    # obs.plot_space(0)
    # obs.plot_space(1)
    # obs.plot_space(2)
    # obs.plot_space(3)
    # obs.plot_space(4)
    # obs.plot_features(5)
    # obs.env.get_cell([5, 5, 5])

    obs.train()
