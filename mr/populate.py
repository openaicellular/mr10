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

import pandas as pd
from influxdb import DataFrameClient
import datetime
import tensorflow as tf
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

import os
import sys
#sys.path.append('C:/Users/Mohammadreza/Desktop/My Class/Proj-DC/My Works/Scheduling/xApp/mr7-main')
sys.path.append('.')

from mr import mobile_env
from mobile_env.handlers.central import MComCentralHandler
from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment

# predefined small scenarios
from mobile_env.scenarios.small import MComSmall


# easy access to the default configuration
# MComSmall.default_config()

# env = gym.make("mobile-small-central-v0")

# num_states = 7
# print("Size of State Space ->  {}".format(num_states))
# num_whole_states = 35
# print("Size of Whole State Space ->  {}".format(num_whole_states))
# num_actions = 4
# print("Size of Action Space ->  {}".format(num_actions))
# num_ues = 5
# upper_bound = env.NUM_STATIONS
# lower_bound = 0
# print("Max Value of Action ->  {}".format(upper_bound))
# print("Min Value of Action ->  {}".format(lower_bound))

# gamma = 0.99  # Discount factor for past rewards
# max_steps_per_episode = 50
# eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


class INSERTDATA:
    print('///////////////enter INSERTDATA class in populate/////////////')

    def __init__(self):
        print('///////enter insert init////////')
        host = 'ricplt-influxdb.ricplt'
        #host = 'localhost'
        print('host=', host)
        #self.client = DataFrameClient(host, '8086', 'root', 'root')
        self.client = DataFrameClient(host, '8086')
        print('self.client=', self.client)
        #print('self.client.get_list_database()=',self.client.get_list_database())
        self.dropdb('RANData')
        print('/////pass dropdb(RANData)////////')
        self.createdb('RANData')
        print('/////pass creatdb(RANData)////////')

    def createdb(self, dbname):
        print('///////enter insert createdb//////////')
        print("Create database: " + dbname)
        self.client.create_database(dbname)
        print('self.client.get_list_database()=', self.client.get_list_database())
        self.client.switch_database(dbname)

    def dropdb(self, dbname):
        print('//////////enter insert dropdb/////////')
        print("DROP database: " + dbname)
        self.client.drop_database(dbname)

    def dropmeas(self, measname):
        print('//////////enter insert dropmeas/////////////')
        print("DROP MEASUREMENT: " + measname)
        self.client.query('DROP MEASUREMENT '+measname)
        print('elf.client.query(DROP MEASUREMENT +measname)=', elf.client.query('DROP MEASUREMENT '+measname))

# def time(df):
#     print('///////////////enter def time//////////////')
#     df.index = pd.date_range(start=datetime.datetime.now(), freq='10ms', periods=len(df))
#     #df.index = pd.to_numeric(pd.date_range(start=datetime.datetime.now(), freq='10ms', periods=len(df)))
#     print('df.index=',df.index)
#     print(df)
#     print('df[0]=', df[0])
#     print('df[35]=', df[35])
#     print('df[36]=', df[36])
#     #print(df['state'])
#     #print('lambda x: str(x)=', lambda x: str(x))
#     #df['state'] = df['state'].apply(lambda x: str(x))
#     df[0] = df[0].apply(lambda x: str(x))
#     print('df=', df)
#     print('df[0]=', df[0])
#     print('df[35]=', df[35])
#     print('df[36]=', df[36])
#     return df

# def populatedb_T0():
#     print('/////////////enter def populatedb_T0()///////////')
#     #data = pd.read_csv('C:/Users/Mohammadreza/Desktop/My Class/Proj-DC/My Works/Scheduling/xApp/mr7-main/mr/cells.csv')
    
#     data_T0 = env.reset()
#     print('data_T0 = env.reset()=', data_T0)

#     data_T0 = pd.DataFrame(data_T0)
#     print('data_T0 = pd.DataFrame(data_T0)=',data_T0)

#     data_T0 = data_T0.T
#     print('data_T0 = data_T0.T=', data_T0)
    
#     data = time(data_T0)
#     print('data= time(data_T0)=',data)
    
#     # inintiate connection and create database UEDATA
#     db = INSERTDATA()
#     print('insert data finished, go to write_point')
#     db
#     print('db =', db)
#     db.client.write_points(data, 'RANMeas_T0')
#     print('db.client.write_points(data, RANMeas_T0)=', db.client.write_points(data, 'RANMeas_T0'))
  
#     del data
    
# def populatedb(action):
#     print('/////////////enter def populatedb()///////////')
#     #data = pd.read_csv('C:/Users/Mohammadreza/Desktop/My Class/Proj-DC/My Works/Scheduling/xApp/mr7-main/mr/cells.csv')
    
#     env.reset()
#     #action = env.action_space.sample()
#     obs, reward, done, info =env.step(action)
#     print('obs=', obs)
#     print('reward=', reward)
#     print('done=', done)
#     print('info=', info)
    
#     obs = obs.tolist()
#     print('obs = obs.tolist()=', obs)
#     data = obs
#     print('data=obs=', data)
#     data.append(reward)
#     print('data.append(reward)=', data)
#     data.append(done)
#     print('data.append(done)=', data)

#     data = pd.DataFrame(data)
#     print('data = pd.DataFrame(data)=',data)

#     data = data.T
#     print('data = data.T=', data)
    
#     data = time(data)
#     print('data= time(data)=',data)
    
#     # inintiate connection and create database UEDATA
#     db = INSERTDATA()
#     print('insert data finished, go to write_point')
#     db
#     print('db =', db)
#     db.client.write_points(data, 'RANMeas')
#     print('db.client.write_points(data, RANMeas)=', db.client.write_points(data, 'RANMeas'))
  
#     del data
    

    

