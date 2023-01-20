

import copy
from multiprocessing import Pipe
import gym
import numpy as np
import pylab
from tensorflow.keras.optimizers import Adam
from tensorboardX import SummaryWriter


from deep_Q_learning.environment import Environment
from deep_Q_learning.models import Actor_Model, Critic_Model


class Agent:
    
    def __init__(self, env_name, model_name=""):
        
        
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 2000 
        self.episode = 0 
        self.max_average = 0 
        self.lr = 0.00025
        self.epochs = 10 
        self.shuffle = True
        self.Training_batch = 512
        
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        
        self.scores_, self.episodes_, self.average_ = [], [], [] 

        
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
 
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)


    def act(self, state):
        
        pred = self.Actor.predict(state)

        low, high = -1.0, 1.0 
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)
        
        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):
        
        
        gamma = 0.99    
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) 
        discounted_r /= (np.std(discounted_r) + 1e-8) 
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        '''
        y_true = np.hstack([advantages, actions, logp_ts])
        
        
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        
        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)

    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            
            
            
            
        else:
            SAVING = ""

        return self.average_[-1], SAVING

    # return the reward of the action
    def reward_function(self, state, action):
        pass

    
    def run_batch(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            for t in range(self.Training_batch):
                self.env.render()
                
                action, logp_t = self.act(state)
                
                next_state, reward, done, _ = self.env.step(action[0])
                
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)
                    
                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])

            self.replay(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break

        self.env.close()


    def run_multiprocesses(self, num_worker = 4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states =        [[] for _ in range(num_worker)]
        next_states =   [[] for _ in range(num_worker)]
        actions =       [[] for _ in range(num_worker)]
        rewards =       [[] for _ in range(num_worker)]
        dones =         [[] for _ in range(num_worker)]
        logp_ts =       [[] for _ in range(num_worker)]
        score =         [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            
            action, logp_pi = self.act(np.reshape(state, [num_worker, self.state_size[0]]))
            
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
                        
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], dones[worker_id], next_states[worker_id], logp_ts[worker_id])

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 100):
        self.load()
        for e in range(101):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.Actor.predict(state)[0]
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()