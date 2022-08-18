#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# ==================================================================================
#       Copyright (c) 2020 China Mobile Technology (USA) Inc. Intellectual Property.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ==================================================================================


import os
import sys
#sys.path.append('C:/Users/Mohammadreza/Desktop/My Class/Proj-DC/My Works/Scheduling/xApp/mr7-main/mr9_github')
sys.path.append('.')
import schedule
import datetime
from zipfile import ZipFile
import json
from os import getenv
from ricxappframe.xapp_frame import RMRXapp, rmr, Xapp
#from mr import sdl

import logging
import numpy as np
import tensorflow as tf
from numpy import zeros, newaxis

from mr import populate
from populate import INSERTDATA
from mr.db import DATABASE, DUMMY
#import mr.populate as populate

from tensorflow import keras
from tensorflow.keras import layers
import gym

import numpy as np
import pandas as pd
import statistics
from statistics import mean
import matplotlib.pyplot as plt
import IPython
from IPython import display

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense  
from tensorflow.keras.layers import Activation  
from tensorflow.keras.optimizers import Adam

from mr import mobile_env
from mobile_env.handlers.central import MComCentralHandler
from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.scenarios.small import MComSmall


# In[ ]:



MComSmall.default_config()

env = gym.make("mobile-small-central-v0")

num_states = 7
print("Size of State Space ->  {}".format(num_states))
num_whole_states = 35
print("Size of Whole State Space ->  {}".format(num_whole_states))
num_actions = 4
print("Size of Action Space ->  {}".format(num_actions))
num_ues = 5
upper_bound = env.NUM_STATIONS
lower_bound = 0
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
# Configuration parameters for the whole setup
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 50
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# num_inputs = 4
# num_actions = 2
num_hidden1 = 64
num_hidden2 = 128

inputs = layers.Input(shape=(num_states,))
common1 = layers.Dense(num_hidden1, activation="relu")(inputs)
common2 = layers.Dense(num_hidden2, activation="relu")(common1)
action = layers.Dense(num_actions, activation="softmax")(common2)
critic = layers.Dense(1, activation="linear")(common2)

model = keras.Model(inputs=inputs, outputs=[action, critic])

print('model.summary in get_actor',model.summary())


xapp = None
pos = 0
RAN_data = None
rmr_xapp = None

class UENotFound(BaseException):
    pass
class CellNotFound(BaseException):
    pass

def post_init(self):
    print('///////enter def post_init__/////////////////')
    """
    Function that runs when xapp initialization is complete
    """
    self.def_hand_called = 0
    self.traffic_steering_requests = 0


def handle_config_change(self, config):
    print('////////enter def handle_config_change//////////////')
    """
    Function that runs at start and on every configuration file change.
    """
    self.logger.debug("handle_config_change: config: {}".format(config))


def default_handler(self, summary, sbuf):
    print('/////////enter def default_handler///////////////')
    """
    Function that processes messages for which no handler is defined
    """
    self.def_hand_called += 1
    print('self.def_hand_called += 1=', self.def_hand_called)
    self.logger.warning("default_handler unexpected message type {}".format(summary[rmr.RMR_MS_MSG_TYPE]))
    self.rmr_free(sbuf)


def entry():
    print('////////////enter def entry///////////////')
    """  Read from DB in an infinite loop and run prediction every second
      TODO: do training as needed in the future
    """
    job = schedule.every(1).seconds.do(RL)
    print('/////////pass 1 entry schedule.every(1).seconds.do(run_prediction, self)/////')
    RL()
    #while True:
        #print('////while True in entry/////') 
        #schedule.run_pending()        
        
def RL():
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    actions_probs_history = []
    action_probs_history_test = []
    critic_value_history = []
    critic_value_history_test = []
    rewards_history = []
    reward_history_for_plot = []
    running_rewards_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    iteration = 0
    utility_history = []
    episode_utility = 0
    episode_utility_history = []
    mean_utility_history = []
    actor_loss_history = []
    critic_loss_history = []
    critic_loss_history_test = []
    loss_value_history_whole = []

    while True:  # Run until solved
        state = env.reset()
        print('tensor state in while True=', state)
        episode_reward = 0

        with tf.GradientTape() as tape:

            for timestep in range(1, max_steps_per_episode):
                print('timestep=', timestep)
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                prev_state = state
                #print('prev_state in main=', prev_state)

                state_per_user = [0]*num_states
                #state_per_user = []

                for index in range (num_ues):

                    state_per_user[index] = state[(index*num_states):((index*num_states)+num_states)]
                    #state_per_user.append(state[(index*num_states):((index*num_states)+num_states)])
                    #print('state_per_user[index]=', state_per_user[index])


                state_all_users = []
                for i in range (num_ues):
                    state_all_users.append(state_per_user[:][i])
                #print('state_all_users=', state_all_users)
                state_all_users = tf.convert_to_tensor(state_all_users)    
                #print('tf.convert_to_tensor(state_all_users)=', state_all_users)

                action_probs, critic_value = model(state_all_users)
                #print('action_probs=', action_probs)
                #print('critic_value=', critic_value)
                #print('critic_value[:, 0]=', critic_value[:, 0])
                critic_value_history.append(critic_value[:, 0])
                #print('critic_value_history.append(critic_value[0, 0])', critic_value_history)
                critic_value_history_test.append(critic_value)
                #print('critic_value_history_test.append(critic_value)=', critic_value_history_test)



                # Sample action from action probability distribution
                #print('paaaaaaaaaaasssseeeddddddddddd')
                actions = [0]*num_ues
                action_probs_per_action = []
                p_whole = []
                for i in range(num_ues):
                    #print('i=', i)
                    #print('action_probs[i]=', action_probs[i])
                    #print('action_probs[i,:]=', action_probs[i,:])
                    #print('p=np.squeeze(action_probs[i,:])=', np.squeeze(action_probs[i,:]))
                    #print('tf.squeeze(action_probs[i,:])=', tf.squeeze(action_probs[i,:]))
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs[i]))

                    #print('action=', action)
                    actions[i]=action
                    #print('actions=', actions)
                    action_probs_per_action.append(action_probs[i, action])
                    #print('action_probs_per_action.append(action_probs[i, action])=', action_probs_per_action)

                    #print('tf.math.log(action_probs[i, action])=', tf.math.log(action_probs[i, action]))
                    action_probs_history.append(tf.math.log(action_probs[i, action]))
                    #print('action_probs_history.append(tf.math.log(action_probs[i, action]))=', action_probs_history)

                #print('tf.math.log(action_probs_per_action)=', tf.math.log(action_probs_per_action))
                action_probs_history_test.append(tf.math.log(action_probs_per_action))
                #print('action_probs_history_test.append(tf.math.log(action_probs_per_action))=',action_probs_history_test)
                actions_probs_history.append(action_probs_history)
                #print('actions_probs_history=', actions_probs_history)
                actions_tensor = tf.convert_to_tensor(actions) 
                print('actions_tensor=', actions_tensor)
                actions = np.asarray(actions, dtype=np.int64)
                #print('actions_a_array=', actions_a)


                # Apply the sampled action in our environment
                
                state, reward, done = connectdb(actions)
                #state, reward, done, info = env.step(actions)
                #network_reward = self.utilities_scaled_float_mean
                #print('state=state, reward, done = connectdb(actions)=', state)
                state = np.array(state, dtype='float32')
                #print('state = np.array(state)=', state)
                print('reward:connectdb(actions)=', reward)
                #print('network_reward=', network_reward)
                print('done=env.step(actions)=', done)
                #print('info=env.step(actions)=', info)
                #print('utility=env.step(actions)=', utility)
                rewards_history.append(reward)
                reward_history_for_plot.append(reward)
                #utility_history.append(utility)
                #print('rewards_history.append(reward)=', rewards_history)
                episode_reward += reward
                #print('episode_reward += reward=', episode_reward)
                episode_reward_history.append(episode_reward)
                #episode_utility +=utility
                #episode_utility_history.append(episode_utility)


                if done:

                    break
            
            print('if done break out of loop')
            #schedule.cancel_job('RL')
            #schedule.clear()
            
            iteration +=1
            # load all tracked results as pandas data frames
            scalar_results_1 = env.monitor.load_results()

            # show general specific results
            #scalar_results_1.head()

            mean_utility_history.append(scalar_results_1['mean utility'].tolist())

            # Update running reward to check condition for solving

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            #print('running_reward = 0.05 * epi=', running_reward)
            running_rewards_history.append(running_reward)
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            #print('returns=', returns)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            #print('returns=(returns - np.mean(returns)) / (np.st', returns)
            returns = returns.tolist()
            #print('returns.tolist()=', returns)

            # Calculating loss values to update our network
            history = zip(action_probs_history_test, critic_value_history, returns, critic_value_history_test)
            #print('history = zip(action_probs_history, critic_value_history, returns)=', history)
            actor_losses = []
            critic_losses = []
            critic_losses_test = []
            for log_prob, value, ret, value_test in history:



                #print('log_prob=', log_prob)
                #print('value=', value)
                #print('ret=', ret)
                diff = ret - value
                #print('diff in for loop of history=', diff)

                #print('diff.mul(log_prob)=', tf.multiply(-log_prob,diff))
                #print('-log_prob * diff=', -log_prob*diff)
                actor_losses.append(-log_prob*diff)  # actor loss
                #actor_losses.append(tf.multiply(-log_prob,diff))  # actor loss
                actor_loss_history.append(actor_losses)
                #print('actor_losses.append(-log_prob * diff)=', actor_losses)

   

                #print('value_test=', value_test)
                #print('tf.expand_dims(value_test, 0)=', tf.expand_dims(value, 0))
                #print('huber_loss(tf.expand_dims(value_test, 0), tf.expand_dims(ret, 0))=', huber_loss(value_test, tf.expand_dims(ret, 0)))
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )
                critic_loss_history.append(critic_losses)
                #print('critic_losses=', critic_losses)





            # Backpropagation
            loss_value_history = []
            grads_history = []
            #print('actor_losses=', actor_losses)
            #print('critic_losses=', critic_losses)
            #print('sum(actor_losses)=', sum(actor_losses))
            #print('sum(critic_losses)=', sum(critic_losses))
            loss_value = sum(actor_losses) + sum(critic_losses)
            #print('loss_value = sum(actor_losses) + sum(critic_losses)=', loss_value)
            loss_value_history.append(loss_value)
            loss_value_history_whole.append(loss_value)
            #print('loss_value for backprpagation=', loss_value)
            grads = tape.gradient(loss_value, model.trainable_variables)
            #print('grads = tape.gradient(loss_value, model.trainable_variables)=', grads)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            grads_history.append(grads)       
            #print('iteration=', iteration)

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        episode_count += 1
        print('episode_count += 1=', episode_count)
        if episode_count % 10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        scalar_results_2 = env.monitor.load_results()
    #     if scalar_results_2['mean datarate'].mean() > 20 and scalar_results_2['mean datarate'].max() < 130:



        if episode_count > 1:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            print("Solved at episode {}!".format(episode_count))
            print("Solved at episode {}!".format(episode_count))

            scalar_results_2 = env.monitor.load_results()

#             plt.plot(reward_history_for_plot)
#             plt.xlabel("episode_count")
#             plt.ylabel("reward_history_for_plot")
#             plt.show()    

#             plt.plot(running_rewards_history)
#             plt.xlabel("episode_count")
#             plt.ylabel("running_rewards_history")
#             plt.show()

#             plt.plot(episode_reward_history)
#             plt.xlabel("episode_count")
#             plt.ylabel("episode_reward")
#             plt.show()

#             plt.plot(loss_value_history)
#             plt.xlabel("iteration")
#             plt.ylabel("loss_value_history")
#             plt.show()

#             plt.plot(utility_history)
#             plt.xlabel("episode_count")
#             plt.ylabel("utility_history")
#             plt.show()

#             plt.plot(episode_utility_history)
#             plt.xlabel("episode_count")
#             plt.ylabel("episode_utility_history")
#             plt.show()


            break

        


def connectdb(action):
    print('////////////////////enter def connectdb///////////////////')
    # Create a connection to InfluxDB if thread=True, otherwise it will create a dummy data instance
    global db
    global RAN_data
    
    print('//////enter else= populate.populate()////////////////')  
    populatedb(action)  # temporary method to populate db, it will be removed when data will be coming through KPIMON to influxDB

    print('////came back from populate to connectdb.else:, db=DATABASE(RANData)///////')
    db = DATABASE('RANData')
    print('////came back from db.DATABASE-init to connectdb.else///////')
    print('db =  DATABASE(RANData) =', db) 
    db.read_data("RANMeas")
    print('////came back from db.DATABASE-read-data to connectdb.else///////')
    print('db.read_data("RANMeas")=', db.read_data("RANMeas"))
    RAN_data = db.data.values.tolist()  # needs to be updated in future when live feed will be coming through KPIMON to influxDB
    print('RAN_data = db.data.values.tolist()=', RAN_data)
    obs = [0] * 35
    #print('obs=', obs)
    obs[0:2] = RAN_data[0][0:2]
    #print('obs=', obs)
    obs[2] = RAN_data[0][12]
    #print('obs=', obs)
    obs[3] = RAN_data[0][23]
    #print('obs=', obs)
    obs[4:10] = RAN_data[0][31:37]
    #print('obs=', obs)
    obs[10:20] = RAN_data[0][2:12]
    #print('obs=', obs)
    obs[20:29] = RAN_data[0][13:23]
    #print('obs=', obs)
    obs[29:35] = RAN_data[0][24:29]
    print('obs=', obs)

    reward = RAN_data[0][29]
    done = RAN_data[0][30]

    #print('RAN_data:, RAN_data)
    print('///////connectdb finished go to start//////')
    return obs, reward, done
 


def start(thread=False):
 
    print('////////////////entered Starrrrrrrrrrrt///////////////////')
    """
    This is a convenience function that allows this xapp to run in Docker
    for "real" (no thread, real SDL), but also easily modified for unit testing
    (e.g., use_fake_sdl). The defaults for this function are for the Dockerized xapp.
    """
    global xapp

    #fake_sdl = getenv("USE_FAKE_SDL", None)
    #xapp = Xapp(entrypoint=entry, rmr_port=4560, use_fake_sdl=False)
    #print('xapp = Xapp(entrypoint=entry, rmr_port=4560, use_fake_sdl=fake_sdl)=', xapp)
  
    use_fake_sdl=False
    rmr_port=4560
    entry()
    
    #xapp.run()


def stop():
    print('/////////////enter def stop//////////////////')      
    """
    can only be called if thread=True when started
    """
    xapp.stop()


def get_stats():
    print('//////////////////enter def get_stats()////////////////////')
    """
    hacky for now, will evolve
    """
    print('DefCalled:rmr_xapp.def_hand_called=', rmr_xapp.def_hand_called)
    print('SteeringRequests:rmr_xapp.traffic_steering_requests=', rmr_xapp.traffic_steering_requests) 
    return {"DefCalled": rmr_xapp.def_hand_called,
            "SteeringRequests": rmr_xapp.traffic_steering_requests}



def time(df):
    print('///////////////enter def time//////////////')
    df.index = pd.date_range(start=datetime.datetime.now(), freq='10ms', periods=len(df))
    #df.index = pd.to_numeric(pd.date_range(start=datetime.datetime.now(), freq='10ms', periods=len(df)))
    #print('df.index=',df.index)
    print(df)
    #print('df[0]=', df[0])
    #print('df[35]=', df[35])
    #print('df[36]=', df[36])
    #print(df['state'])
    #print('lambda x: str(x)=', lambda x: str(x))
    #df['state'] = df['state'].apply(lambda x: str(x))
    df[0] = df[0].apply(lambda x: str(x))
    #print('df=', df)
    #print('df[0]=', df[0])
    #print('df[35]=', df[35])
    #print('df[36]=', df[36])
    return df

def populatedb_T0():
    print('/////////////enter def populatedb_T0()///////////')
    #data = pd.read_csv('C:/Users/Mohammadreza/Desktop/My Class/Proj-DC/My Works/Scheduling/xApp/mr7-main/mr/cells.csv')
    
    data_T0 = env.reset()
    print('data_T0 = env.reset()=', data_T0)

    data_T0 = pd.DataFrame(data_T0)
    print('data_T0 = pd.DataFrame(data_T0)=',data_T0)

    data_T0 = data_T0.T
    print('data_T0 = data_T0.T=', data_T0)
    
    data = time(data_T0)
    print('data= time(data_T0)=',data)
    
    # inintiate connection and create database UEDATA
    db = INSERTDATA()
    print('insert data finished, go to write_point')
    db
    print('db =', db)
    db.client.write_points(data, 'RANMeas_T0')
    print('db.client.write_points(data, RANMeas_T0)=', db.client.write_points(data, 'RANMeas_T0'))
  
    del data
    
def populatedb(action):
    print('/////////////enter def populatedb()///////////')
    #data = pd.read_csv('C:/Users/Mohammadreza/Desktop/My Class/Proj-DC/My Works/Scheduling/xApp/mr7-main/mr/cells.csv')
    
    env.reset()
    #action = env.action_space.sample()
    obs, reward, done, info =env.step(action)
    #print('obs=', obs)
    #print('reward=', reward)
    #print('done=', done)
    #print('info=', info)
    
    obs = obs.tolist()
    #print('obs = obs.tolist()=', obs)
    data = obs
    #print('data=obs=', data)
    data.append(reward)
    #print('data.append(reward)=', data)
    data.append(done)
    #print('data.append(done)=', data)

    data = pd.DataFrame(data)
    #print('data = pd.DataFrame(data)=',data)

    data = data.T
    #print('data = data.T=', data)
    
    data = time(data)
    #print('data= time(data)=',data)
    
    # inintiate connection and create database UEDATA
    db = INSERTDATA()
    #print('insert data finished, go to write_point')
    db
    #print('db =', db)
    db.client.write_points(data, 'RANMeas')
    #print('db.client.write_points(data, RANMeas)=', db.client.write_points(data, 'RANMeas'))
  
    del data
    


def mr_req_handler(self, summary, sbuf):
    print('///////////enter def mr_req handler/////////////')
    """
    This is the main handler for this xapp, which handles load prediction requests.
    This app fetches a set of data from SDL, and calls the predict method to perform
    prediction based on the data

    The incoming message that this function handles looks like:
        {"UEPredictionSet" : ["UEId1","UEId2","UEId3"]}
    """
    #self.traffic_steering_requests += 1
    # we don't use rts here; free the buffer
    self.rmr_free(sbuf)

    ue_list = []
    try:
        print('////enter first try in mr_req_handler////')
        print('rmr.RMR_MS_PAYLOAD=', rmr.RMR_MS_PAYLOAD)
        print('summary[rmr.RMR_MS_PAYLOAD]=', summary[rmr.RMR_MS_PAYLOAD])
        req = json.loads(summary[rmr.RMR_MS_PAYLOAD])  # input should be a json encoded as bytes
        print('req = json.loads(summary[rmr.RMR_MS_PAYLOAD])=', req)
        ue_list = req["UEPredictionSet"]
        print('ue_list=req["UEPredictionSet"] =', ue_list)
        self.logger.debug("mr_req_handler processing request for UE list {}".format(ue_list))
    except (json.decoder.JSONDecodeError, KeyError):
        print('////enter first except in mr_req_handler////')
        self.logger.warning("mr_req_handler failed to parse request: {}".format(summary[rmr.RMR_MS_PAYLOAD]))
        return
    print('ue_list mr_req_handler aftr 1st try=', ue_list)
    # iterate over the UEs, fetches data for each UE and perform prediction
    for ueid in ue_list:
        try:
            print('////enter second try in mr_req_handler////')
            uedata = sdl.get_uedata(self, ueid)
            print('uedata = sdl.get_uedata(self, ueid)=', uedata)
            predict(self, uedata)
            print('predict(self, uedata)=', predict(self, uedata))
        except UENotFound:
            print('////enter second except in mr_req_handler////')
            print('enter UENotFound in mr_req_handler')
            self.logger.warning("mr_req_handler received a TS Request for a UE that does not exist!")    
    
 

