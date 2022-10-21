#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
import click
import torch
from garage.torch import set_gpu_mode

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import task_sampler, MetaEvaluator, OnlineMetaEvaluator, Snapshotter
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.trainer import Trainer
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.policies import GaussianTransformerPolicy, GaussianTransformerEncoderPolicy, GaussianMemoryTransformerPolicy, GaussianMLPPolicy, BetaTransformerEncoderPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from prettytable import PrettyTable

import gym
from gym import spaces
from gym.envs.registration import register
import torch as t
import math
import numpy as np

from utils import animate_fourier

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params




class FourierEnv(gym.Env):
    metadata = {
        "render.modes": None,
    }
    
    def __init__(self, task=None, x_dim=2, n_cos=100):
        # set action/observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(x_dim,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(x_dim+1,))
        self.eta = None
        self.x_dim = x_dim
        self.n_cos = n_cos

        if task is None:
            self._task = self.sample_tasks(1)
        else:
            self._task = task


    def step(self, action):
        # action = 1 / (1 + np.exp(-5*action))
        # action = np.tanh(action)
        # action = np.clip(action, 0, 1)
        l = self._task['l']
        wavevectors = self._task['wavevectors'] / l
        phases = self._task['phases']
        rand = self._task['rand']

        freq = action @ wavevectors
        y = np.cos(freq + phases)
        obs = math.sqrt(2)*y @ rand/math.sqrt(self.n_cos)
        # reward = -obs
        reward = np.maximum(self.eta - obs, 0)
        # reward -= 10 * np.maximum(np.abs(action-0.5)-0.5,0).mean()
        # reward -= 10 * np.maximum(np.abs(action-1),0).mean()
        self.eta = np.minimum(obs, self.eta)
        done = False
        
        return np.concatenate([action,obs], axis=-1), reward, done, {}
      
    def reset(self):
        l = self._task['l']
        wavevectors = self._task['wavevectors'] / l
        phases = self._task['phases']
        rand = self._task['rand']

        action = self.action_space.sample()*0
        freq = action @ wavevectors
        y = np.cos(freq + phases)
        obs = math.sqrt(2)*y @ rand/math.sqrt(self.n_cos)

        self.eta = obs
        return np.concatenate([action,obs], axis=-1)   # might need to concat with actions

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, float]]: A list of "tasks," where each task is a
                dictionary containing a single key, "velocity", mapping to a
                value between 0 and 3.

        """
        l_array = 1/t.distributions.gamma.Gamma(83,12.3).sample([num_tasks]).detach().numpy()
        phases_array = np.random.rand(num_tasks, self.n_cos) * math.tau
        wavevectors_array = np.random.randn(num_tasks, self.x_dim, self.n_cos)  #/self.l
        rand_array = np.random.randn(num_tasks, self.n_cos,1)
        tasks = [{'l': l, 'phases':phases, 'wavevectors':wavevectors, 'rand':rand} for (l,phases,wavevectors,rand) in zip(l_array, phases_array, wavevectors_array, rand_array)]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, float]): A task (a dictionary containing a single
                key, "velocity", usually between 0 and 3).

        """
        self._task = task

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instances dictionary to be pickled.

        """
        return dict(task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(task=state['task'])



@click.command()
@click.option('--env_name', default="FourierEnv")
@click.option('--seed', default=1)
@click.option('--max_episode_length', default=10)
@click.option('--meta_batch_size', default=16)
@click.option('--n_epochs', default=1000)
@click.option('--episode_per_task', default=2)
@click.option('--wm_embedding_hidden_size', default=32)
@click.option('--n_heads', default=16)
@click.option('--d_model', default=128)
@click.option('--layers', default=4)
@click.option('--dropout', default=0.0)
@click.option('--wm_size', default=10)
@click.option('--em_size', default=10)
@click.option('--dim_ff', default=128)
@click.option('--discount', default=1)
@click.option('--gae_lambda', default=0.8)
@click.option('--lr_clip_range', default=0.1)
@click.option('--policy_lr', default=3e-5)
@click.option('--vf_lr', default=2.5e-4)
@click.option('--minibatch_size', default=256)
@click.option('--max_opt_epochs', default=10)
@click.option('--center_adv', is_flag=True)
@click.option('--positive_adv', is_flag=True)
@click.option('--policy_ent_coeff', default=0.02)
@click.option('--use_softplus_entropy', is_flag=True, default=True)
@click.option('--stop_entropy_gradient', is_flag=True)
@click.option('--entropy_method', default='regularized')
@click.option('--share_network', is_flag=True) 
@click.option('--architecture', default="Encoder")
@click.option('--policy_head_input', default="latest_memory")
@click.option('--dropatt', default=0.0)
@click.option('--attn_type', default=1)
@click.option('--pre_lnorm', is_flag=True, default=True)
@click.option('--init_params', is_flag=True, default=True)
@click.option('--gating', default="residual")
@click.option('--init_std', default=1.0)
@click.option('--learn_std', is_flag=True, default=True)
@click.option('--policy_head_type', default="Default")
@click.option('--policy_lr_schedule', default="no_schedule")
@click.option('--vf_lr_schedule', default="no_schedule")
@click.option('--decay_epoch_init', default=500)
@click.option('--decay_epoch_end', default=1000)
@click.option('--min_lr_factor', default=0.1)
@click.option('--tfixup', is_flag=True, default=True)
@click.option('--remove_ln', is_flag=True, default=True)
@click.option('--recurrent_policy', is_flag=True, default=False)
@click.option('--pretrained_dir', default=None)
@click.option('--pretrained_epoch', default=4980)
@click.option('--output_weights_scale', default=1.0)
@click.option('--normalized_wm', is_flag=True, default=False)
@click.option('--annealing_std', is_flag=True)
@click.option('--min_std', default=1e-6)
@click.option('--gpu_id', default=0)
@wrap_experiment(snapshot_mode='gap', snapshot_gap=50, log_dir='/user/home/ad20999/transformers-metarl/data/local/experiment')
def transformer_ppo_fourier(ctxt, env_name, seed, max_episode_length, meta_batch_size,
                        n_epochs, episode_per_task,
                        wm_embedding_hidden_size, n_heads, d_model, layers, dropout,
                        wm_size, em_size, dim_ff, discount, gae_lambda, lr_clip_range, policy_lr,
                        vf_lr, minibatch_size, max_opt_epochs, center_adv, positive_adv, 
                        policy_ent_coeff, use_softplus_entropy, stop_entropy_gradient, entropy_method,
                        share_network, architecture, policy_head_input, dropatt, attn_type,
                        pre_lnorm, init_params, gating, init_std, learn_std, policy_head_type,
                        policy_lr_schedule, vf_lr_schedule, decay_epoch_init, decay_epoch_end, min_lr_factor,
                        recurrent_policy, tfixup, remove_ln, pretrained_dir, pretrained_epoch, 
                        output_weights_scale, normalized_wm, annealing_std, min_std, gpu_id):
    """Train PPO with HalfCheetah environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)

    policy = None
    value_function = None
    if pretrained_dir is not None:
        snapshotter = Snapshotter()
        data = snapshotter.load(pretrained_dir, itr=pretrained_epoch)
        policy = data['algo'].policy
        value_function = data['algo'].value_function

    trainer = Trainer(ctxt)
    env_class = FourierEnv
    tasks = task_sampler.SetTaskSampler(
        env_class,
        wrapper=lambda env, _: RL2Env(
            GymEnv(env, max_episode_length=max_episode_length)))

    env_spec = RL2Env(
        GymEnv(env_class(),
                max_episode_length=max_episode_length)).spec

    if annealing_std:
        annealing_rate = (min_std/init_std) ** (3.0 / (meta_batch_size * 2 * n_epochs)) # reach min step at 2/3 * n_epochs
    else:
        annealing_rate = 1.0

    if architecture == "Encoder":
        # policy = GaussianTransformerEncoderPolicy(name='policy',
        #                             # mlp_output_nonlinearity=torch.sigmoid,
        #                             env_spec=env_spec,
        #                             encoding_hidden_sizes=(wm_embedding_hidden_size,),
        #                             nhead=n_heads,
        #                             d_model=d_model,
        #                             num_encoder_layers=layers,
        #                             dropout=dropout,
        #                             obs_horizon=wm_size,
        #                             dim_feedforward=dim_ff,
        #                             policy_head_input=policy_head_input,
        #                             policy_head_type=policy_head_type,
        #                             tfixup=tfixup,
        #                             remove_ln=remove_ln,
        #                             init_std=init_std,
        #                             learn_std=learn_std,
        #                             min_std=min_std,
        #                             annealing_rate=annealing_rate,
        #                             mlp_output_w_init= lambda x: torch.nn.init.xavier_uniform_(x, gain=output_weights_scale),
        #                             normalize_wm=normalized_wm,
        #                             recurrent_policy=recurrent_policy) if policy is None else policy

        policy = BetaTransformerEncoderPolicy(name='policy',
                                    # mlp_output_nonlinearity=torch.sigmoid,
                                    env_spec=env_spec,
                                    encoding_hidden_sizes=(wm_embedding_hidden_size,),
                                    nhead=n_heads,
                                    d_model=d_model,
                                    num_encoder_layers=layers,
                                    dropout=dropout,
                                    obs_horizon=wm_size,
                                    dim_feedforward=dim_ff,
                                    policy_head_input=policy_head_input,
                                    policy_head_type=policy_head_type,
                                    tfixup=tfixup,
                                    remove_ln=remove_ln,
                                    init_std=init_std,
                                    learn_std=learn_std,
                                    min_std=min_std,
                                    annealing_rate=annealing_rate,
                                    mlp_output_w_init= lambda x: torch.nn.init.xavier_uniform_(x, gain=output_weights_scale),
                                    normalize_wm=normalized_wm,
                                    recurrent_policy=recurrent_policy) if policy is None else policy

    elif architecture == "Transformer":         
        policy = GaussianTransformerPolicy(name='policy',
                                    env_spec=env_spec,
                                    encoding_hidden_sizes=(wm_embedding_hidden_size,),
                                    nhead=n_heads,
                                    d_model=d_model,
                                    num_decoder_layers=layers,
                                    num_encoder_layers=layers,
                                    dropout=dropout,
                                    obs_horizon=wm_size,
                                    hidden_horizon=em_size,
                                    dim_feedforward=dim_ff)
    elif architecture == "MemoryTransformer":
        policy = GaussianMemoryTransformerPolicy(name='policy',
                                    env_spec=env_spec,
                                    encoding_hidden_sizes=(wm_embedding_hidden_size,),
                                    nhead=n_heads,
                                    d_model=d_model,
                                    num_encoder_layers=layers,
                                    dropout=dropout,
                                    dropatt=dropatt,
                                    obs_horizon=wm_size,
                                    mem_len=em_size,
                                    dim_feedforward=dim_ff,
                                    attn_type=attn_type,
                                    pre_lnorm=pre_lnorm,
                                    init_params=init_params,
                                    gating=gating,
                                    init_std=init_std,
                                    learn_std=learn_std,
                                    policy_head_type=policy_head_type,
                                    policy_head_input=policy_head_input)
                                    

    # count_parameters(policy)

    base_model = policy if share_network else None

    value_function = GaussianMLPValueFunction(env_spec=env_spec,
                                              base_model=base_model,
                                              hidden_sizes=(64, 64),
                                              learn_std=learn_std,
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    # count_parameters(value_function)

    meta_evaluator = OnlineMetaEvaluator(test_task_sampler=tasks,
                                        n_test_tasks=1,
                                        n_test_episodes=1,
                                        worker_class=RL2Worker,
                                        worker_args=dict(n_episodes_per_trial=2))
    # meta_evaluator = None

    steps_per_epoch = max_opt_epochs * (max_episode_length * episode_per_task * meta_batch_size) // minibatch_size
    # steps_per_epoch = 1

    algo = RL2PPO(meta_batch_size=meta_batch_size,
                    task_sampler=tasks,
                    env_spec=env_spec,
                    policy=policy,
                    value_function=value_function,
                    episodes_per_trial=episode_per_task,
                    discount=discount,
                    gae_lambda=gae_lambda,
                    lr_clip_range=lr_clip_range,
                    policy_lr=policy_lr,
                    vf_lr=vf_lr,
                    minibatch_size=minibatch_size,
                    max_opt_epochs=max_opt_epochs,
                    use_softplus_entropy=use_softplus_entropy,
                    stop_entropy_gradient=stop_entropy_gradient,
                    entropy_method=entropy_method,
                    policy_ent_coeff=policy_ent_coeff,
                    center_adv=center_adv,
                    positive_adv=positive_adv,
                    meta_evaluator=meta_evaluator,
                    policy_lr_schedule=policy_lr_schedule,
                    vf_lr_schedule=vf_lr_schedule,
                    decay_epoch_init=decay_epoch_init,
                    decay_epoch_end=decay_epoch_end,
                    min_lr_factor=min_lr_factor,
                    steps_per_epoch=steps_per_epoch,
                    n_epochs=n_epochs,
                    n_epochs_per_eval=100)

    if torch.cuda.is_available() and gpu_id >= 0:
        set_gpu_mode(True, gpu_id)
    else:
        set_gpu_mode(False)
    algo.to()

    trainer.setup(algo,
                    tasks.sample(meta_batch_size),
                    sampler_cls=LocalSampler,
                    n_workers=meta_batch_size,
                    worker_class=RL2Worker,
                    worker_args=dict(n_episodes_per_trial=episode_per_task))

    trainer.train(n_epochs=n_epochs,
                    batch_size=episode_per_task * max_episode_length *
                    meta_batch_size)

    

    animate = animate_fourier()
    animate._data_dict_list = meta_evaluator._data_dict_list
    animate.plot(fname=f'test_{max_episode_length}_{n_epochs}_2')


transformer_ppo_fourier()
