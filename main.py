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

from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
from functools import reduce
import operator
import seaborn as sns
import pandas as pd
import pickle

from torchrl.envs.libs import pettingzoo

torch.set_default_dtype(torch.half)


class Environment:
    def __init__(self, size=(21, 21, 21), diff=1, risk_prob=0.1, death_prob=0.01):
        self.size = size  # Grid size, e.g., (100, 100, 100) for 3D
        self.vol = size[0] * size[1] * size[2]
        self.diff = diff  # A difficulty or variability factor
        self.dim = len(size)
        self.target = (0, 0, 0, 0)  # energy-p, wall-p, risk-p, death-p by volume or by cells, index by 0
        self.isRiskCluster = True

        # main game -> min (time(steps) * total_energy) -> compared with (l2_distance * harmonic_or_mean_or_70_or_max_energy). if agent_score < estimated_score, good. if a_s > e_s, improve. hm7m is param tightened by agent performance
        # minimize energy = main objective
        self.energy = torch.clamp(torch.normal(0.2, 0.5, size=self.size), min=0, max=3)
        # cannot transpose to wall, if slip into wall -> terminate , death ,-r
        self.wall = torch.zeros(size)
        # risk will lead to slip to a random neighbor cell
        self.risk = torch.mul(torch.rand(size), risk_prob)
        # if death will terminal state, negative reward (-r)
        self.death = torch.mul(self.risk, death_prob)

        self.set_diff()
        self.init_properties()
        self.map = torch.add(torch.add(self.energy.clone(), self.wall.clone()),
                             torch.mul(self.death.clone(), self.risk.clone()))

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
        if self.risk[location] > torch.rand(1):
            return True, self.risk[location]
        else:
            return False, self.risk[location]

    def _get_cell(self, location):
        return self.risk[location], self.energy[location], self.wall[location], self.death[location]

    def cell_cost(self, location):
        tuple(self._get_cell(location))


class MiniGame:
    def __init__(self):
        self.env = tictactoe_v3.env(render_mode="human")
        self.env.reset(seed=42)
        # opt adam or RMSPrompt, Adam dynamic gradient may be bad for pomdp

        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()

            if termination or truncation:
                action = None
            else:
                mask = observation["action_mask"]
                # this is where you would insert your policy
                action = self.env.action_space(agent).sample(mask)

            self.env.step(action)
        self.env.close()
        pass


class Agent(ABC):
    def __init__(self, start_pos, end_pos, team, action_size, state_size, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01, memory_size=10000):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.start_pos = start_pos
        self.current_pos = start_pos
        self.request_pos = start_pos
        self.end_pos = end_pos
        self.team = team
        self.history = None
        self.state = None
        self.action = None

    def lyapunov(self):
        return

    def safety_soft(self, state, action):
        return action

    def CLF(self):
        # should assess next worst possible state.
        # Theorm -> agent in a ok state should not die because of a nudge
        # like balancing a bottle, bottle at stable state ss should not fall down when k force is applied leading it to state sa or sb, however sc may be dangerous and sf is prohibited.
        return

    def safety_hard(self, state, action):
        if violate:
            return True

        return False

    def expect(self, state, action):
        return

    @abstractmethod
    def get_action(self, state):
        """
        Get the action to take based on the current state.

        Args:
            state: The current state in the environment.

        Returns:
            An integer representing the chosen action.
        """
        pass

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
        pass

    @abstractmethod
    def train(self):
        """
        Train the agent using experiences stored in memory.
        """
        pass


class NN(nn.Module):
    def __init__(self, input, output, mode=0):
        # https://r-knott.surrey.ac.uk/Fibonacci/fibtable.html
        if mode == 0:
            super(NN, self).__init__()
            self.layer[0] = nn.Linear(input, 610),
            self.layer[1] = nn.SELU(),
            self.layer[2] = nn.Linear(610, 377),
            self.layer[3] = nn.SELU(),
            self.layer[4] = nn.Linear(377, 144),
            self.layer[5] = nn.SELU(),
            self.layer[6] = nn.Linear(144, 233),
            self.layer[7] = nn.SELU(),
            self.layer[8] = nn.Linear(233, output),
            self.layer[9] = nn.Softmax(dim=-1)
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif mode == 1:
            # dummy for random
            pass

    def forward(self, x):
        pass


class PGAAPP(Agent, NN):
    def __init__(self, start_pos, end_pos, team, policy_type):
        pass


class DQN(Agent, NN):
    def __init__(self, start_pos, end_pos, team, policy_type):
        pass


class Observer:
    def __init__(self):
        self.env = Environment()
        self.log = []
        self.MARL = True
        self.LoadMap = False
        self.LoadNavModel = False
        self.LoadPvPModel = False
        self.ResumeNav = False
        self.ResumePvP = False

        # agent = Agent(start_pos=(0, 0), end_pos=(9, 9), environment=self.env, team=1)

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
        print("features.shape is: ", tensors.shape)
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

    def observation(self, location, target, obs=3, k=3):
        # Method to build obs for NAVagent
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
            # builds the relative properties for the NN, eg abs distance from target
            return torch.cat([torch.tensor(location == target).flatten(),
                              torch.tensor(target - location).flatten()])

        return torch.cat([obs_map(location, obs, k), obs_nav(torch.tensor(location), torch.tensor(target))])

    def train(self):
        # trains agents
        # assign Policy, Agent pairs
        #
        return

    def review(self):

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
    obs.plot_features(5)
