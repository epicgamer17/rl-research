import multiprocessing as mp
import numpy as np
import sys
import traceback
from enum import Enum

class Command(Enum):
    RESET = 1
    STEP = 2
    LAST = 3
    AGENT_SELECTION = 4
    CLOSE = 5

def worker(index, env_fn, pipe, auto_reset):
    try:
        env = env_fn()
        while True:
            try:
                cmd, data = pipe.recv()
            except EOFError:
                break # Parent closed the pipe

            if cmd == Command.RESET:
                env.reset()
                pipe.send(env.last())
            elif cmd == Command.STEP:
                env.step(data)
                obs, rew, term, trunc, info = env.last()
                # Expose rewards for all agents
                info["rewards"] = env.rewards.copy() if hasattr(env, "rewards") else {}
                
                if auto_reset and (term or trunc):
                    final_obs = obs
                    final_info = info.copy()
                    final_reward = rew
                    
                    env.reset()
                    # Get new state
                    obs, rew, term, trunc, info = env.last()
                    info["rewards"] = env.rewards.copy() if hasattr(env, "rewards") else {}
                    
                    # Inject final stats into info
                    info['final_observation'] = final_obs
                    info['final_info'] = final_info
                    info['final_reward'] = final_reward
                    # Flag to indicate a reset happened
                    info['episode_reset'] = True
                    
                    # We return the NEW state, but with info about the OLD state end
                    pipe.send((obs, rew, term, trunc, info))
                else:
                    pipe.send((obs, rew, term, trunc, info))
            elif cmd == Command.LAST:
                pipe.send(env.last())
            elif cmd == Command.AGENT_SELECTION:
                pipe.send(env.agent_selection)
            elif cmd == Command.CLOSE:
                env.close()
                pipe.close()
                break
    except Exception as e:
        traceback.print_exc()
        try:
            pipe.send(e)
        except (BrokenPipeError, EOFError):
            pass # Parent is already gone
        pipe.close()

class VectorAECEnv:
    def __init__(self, env_fns, auto_reset=True):
        self.num_envs = len(env_fns)
        self.auto_reset = auto_reset
        self.pipes = []
        self.processes = []
        
        for i, env_fn in enumerate(env_fns):
            parent_pipe, child_pipe = mp.Pipe()
            p = mp.Process(target=worker, args=(i, env_fn, child_pipe, auto_reset))
            p.start()
            self.pipes.append(parent_pipe)
            self.processes.append(p)
            
    def reset(self):
        for pipe in self.pipes:
            pipe.send((Command.RESET, None))
        results = [pipe.recv() for pipe in self.pipes]
        self._check_errors(results)
        
        obs_list, rew_list, term_list, trunc_list, info_list = zip(*results)
        
        self._observations = np.stack(obs_list)
        self._rewards = np.array(rew_list)
        self._terminations = np.array(term_list)
        self._truncations = np.array(trunc_list)
        self._infos = list(info_list)
        
    def step(self, actions):
        """
        Step each environment with the corresponding action.
        actions: list or array of actions of length num_envs
        """
        for i, pipe in enumerate(self.pipes):
            pipe.send((Command.STEP, actions[i]))
            
        results = [pipe.recv() for pipe in self.pipes]
        self._check_errors(results)
        
        obs_list, rew_list, term_list, trunc_list, info_list = zip(*results)
        
        self._observations = np.stack(obs_list)
        self._rewards = np.array(rew_list)
        self._terminations = np.array(term_list)
        self._truncations = np.array(trunc_list)
        self._infos = list(info_list)

    def last(self):
        return (
            self._observations,
            self._rewards,
            self._terminations,
            self._truncations,
            self._infos
        )

    @property
    def rewards(self):
        return self._rewards

    @property
    def terminations(self):
        return self._terminations

    @property
    def truncations(self):
        return self._truncations

    @property
    def infos(self):
        return self._infos
        
    def agent_selection(self):
        for pipe in self.pipes:
            pipe.send((Command.AGENT_SELECTION, None))
        results = [pipe.recv() for pipe in self.pipes]
        self._check_errors(results)
        return results

    def close(self):
        for pipe in self.pipes:
            pipe.send((Command.CLOSE, None))
        for p in self.processes:
            p.join()
            
    def _check_errors(self, results):
        for res in results:
            if isinstance(res, Exception):
                raise res
