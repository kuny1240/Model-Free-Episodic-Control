import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
import os

class Simulator(object):
    def __init__(self):
        # Model Load
        self.cur_path = os.path.abspath(os.path.curdir)
        self.kpi_model = tf.keras.models.load_model(self.cur_path + '/Models/KPI_model_best.h5')
        self.state_scaler = pickle.load(open(self.cur_path + '/Models/state_scaler.pkl','rb'))
        self.action_scaler = pickle.load(open(self.cur_path + '/Models/action_scaler.pkl', 'rb'))
        #__________________________________________________________________________
        self.states = pd.read_csv(self.cur_path + '/Data_Bases/States_data.csv',
                                  index_col=None).values
        self.kpis = pd.read_csv(self.cur_path + '/Data_Bases/KPI_data.csv',
                                index_col=None).values
        self.actions = pd.read_csv(self.cur_path + '/Data_Bases/Action_data.csv',
                                   index_col=None).values
        # Initialize Positions
        self.pos = -1 # Not initialized
        self.state = np.zeros((131,))
        self.action = np.zeros((4,))
        self.random_cols = list(range(15,41)) # these cols change a random gaussian noise with mean as the beginning status

        # Initialize Bases
        self.noise_base = np.zeros((len(self.random_cols),))
        self.action_space = np.array([[-4,4],
                                      [11,22],
                                      [2,12],
                                      [2,30]])

        # Initialize factor
        self.ini_kpi = 0
        self.cur_step = 0
        self.cur_kpi = 0
        self.done = 0
        self.stuck_step = 0

    def instant_reward(self,cur_kpi,pre_kpi):
        return 10 * (pre_kpi - cur_kpi) / pre_kpi

    def reset(self):
        ID = pd.read_csv(self.cur_path + '/Data_Bases/ID.csv')
        self.pos =  random.choice(list(range(ID.shape[0])))
        self.state = self.state = self.states[self.pos:self.pos + 1,:]
        self.ini_kpi = self.cur_kpi = self.kpis[self.pos:self.pos + 1,:]
        if self.ini_kpi == 0 :
            self.ini_kpi = self.cur_kpi = 1
        self.cur_step = 0
        self.done = 0
        self.stuck_step = 0
        print("Now at cell:" + ID.iloc[self.pos,2])
        print("Today is:" + ID.iloc[self.pos,0] +"Initial KPI is: {}".format(self.ini_kpi))
        # basic possible states from existing points
        self.noise_base = self.state[:, self.random_cols] * 0.05 # at most 5% change
        return self.state_scaler.transform(self.state)

    def step(self, action):
        k = np.random.binomial(1, .5, len(self.random_cols))
        k[k == 0] = -1
        self.state[:, self.random_cols] = self.noise_base * np.random.rand(len(self.random_cols)) * k + self.state[:, self.random_cols]

        pred_input = [self.state_scaler.transform(self.state),action]
        pred_kpi = self.kpi_model.predict(pred_input)

        reward = self.instant_reward(pred_kpi,self.cur_kpi)
        if reward < 0.5:
            self.stuck_step += 1
        else:
            self.stuck_step == 0

        self.cur_step += 1

        if self.stuck_step >= 7 or self.cur_step == 21:
            self.done = 1
            reward += 10 * self.instant_reward(pred_kpi,self.ini_kpi)
            print('Episode Ends, current kpi is {}'.format(pred_kpi))

        self.cur_kpi = pred_kpi

        return self.state_scaler.transform(self.state), reward, self.done

    def set_state(self,state):
        self.state = state

class Val_Simulator(object):
    def __init__(self):
        # Model Load
        self.cur_path = os.path.abspath(os.path.curdir)
        self.kpi_model = tf.keras.models.load_model(self.cur_path + '/Models/KPI_model_best.h5')
        self.state_scaler = pickle.load(open(self.cur_path + '/Models/state_scaler.pkl', 'rb'))
        self.action_scaler = pickle.load(open(self.cur_path + '/Models/action_scaler.pkl', 'rb'))
        #__________________________________________________________________________

        self.states =  pd.read_csv(self.cur_path + '/Data_Bases/States_data_val.csv',
                                 index_col=None).values
        self.kpis = pd.read_csv(self.cur_path + '/Data_Bases/KPI_data_val.csv',
                                 index_col=None).values
        self.actions = pd.read_csv(self.cur_path + '/Data_Bases/Action_data_val.csv',
                                 index_col=None).values
        # Initialize Positions
        self.pos = -1 # Not initialized
        self.state = np.zeros((131,))
        self.action = np.zeros((4,))
        self.random_cols = list(range(15,41)) # these cols change a random gaussian noise with mean as the beginning status

        # Initialize Bases
        self.noise_base = np.zeros((len(self.random_cols),))
        self.action_space = np.array([[-4,4],
                                      [11,22],
                                      [2,12],
                                      [2,30]])

        # Initialize factor
        self.ini_kpi = 0
        self.cur_step = 0
        self.cur_kpi = 0
        self.done = 0
        self.stuck_step = 0

    def instant_reward(self,cur_kpi,pre_kpi):
        return 10 * (pre_kpi - cur_kpi) / pre_kpi

    def reset(self):
        ID = pd.read_csv(self.cur_path + '/Data_Bases/ID_val.csv')
        self.pos =  random.choice(list(range(ID.shape[0])))
        self.state = self.states[self.pos:self.pos + 1,:]
        self.action = self.actions[self.pos:self.pos + 1,:]
        self.ini_kpi = self.cur_kpi =  self.kpis[self.pos:self.pos + 1,:]
        if self.ini_kpi == 0 :
            self.ini_kpi = self.cur_kpi = 1

        self.cur_step = 0
        self.done = 0
        self.stuck_step = 0
        print("Now at cell:" + ID.iloc[self.pos,1])
        print("Today is:" + ID.iloc[self.pos,0] +"Initial KPI is: {}".format(self.ini_kpi))
        # basic possible states from existing points
        self.noise_base = self.state[:, self.random_cols] * 0.05  # at most 5% change
        return self.state_scaler.transform(self.state)

    def step(self, action):
        k = np.random.binomial(1, .5, len(self.random_cols))
        k[k == 0] = -1
        self.state[:, self.random_cols] = self.noise_base * np.random.rand(len(self.random_cols)) * k + self.state[:, self.random_cols]

        pred_input = [self.state_scaler.transform(self.state),action]
        pred_kpi = self.kpi_model.predict(pred_input)

        reward = self.instant_reward(pred_kpi,self.cur_kpi)
        if reward < 0.5:
            self.stuck_step += 1
        else:
            self.stuck_step == 0

        self.cur_step += 1

        if self.stuck_step >= 7 or self.cur_step == 21:
            self.done = 1
            reward += 10 * self.instant_reward(pred_kpi,self.ini_kpi)
            print('Episode Ends, current kpi is {}'.format(pred_kpi))

        self.cur_kpi = pred_kpi

        return self.state_scaler.transform(self.state), reward, self.done

    def set_state(self,state):
        self.state = state


if __name__ == "__main__":
    simulator = Simulator()
    state = simulator.reset()
    next_state,reward,done = simulator.step(np.array([[1,1,1,1]]))
    next_state, reward, done = simulator.step(np.array([[0, 0, 0, 0]]))
