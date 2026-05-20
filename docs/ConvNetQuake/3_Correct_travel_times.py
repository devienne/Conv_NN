#!/usr/bin/env python
# coding: utf-8

# # Correct times to travel time
# 
# * This notebook were used to correct the time of the earthquake to the travel time of the seismic wave to the station


from obspy.taup import TauPyModel
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the list of historical earthquake previously downloaded
historical = pd.read_csv('/Users/josea/Desktop/Big_Data/Earthquake_detection/historical.csv')


# for description of 'ak135' model, please refer to ['Kennett et al., 1995]
model = TauPyModel(mod='ak135')

arrival_times = []

for i in range(len(historical)):
    arrivals = modl.get_travel_times_geo(source_depth_in_km=historical.loc[i]['depth_km'],
                                      source_latitude_in_deg=historical.loc[i]['latitude'],
                                      source_longitude_in_deg=historical.loc[i]['longitude'],
                                      receiver_latitude_in_deg=35.711956, # lat of seismic station
                                      receiver_longitude_in_deg=-97.283646, # lon of seismic station
                                  phase_list=['p'])
try:
    arrivals_times.append(round(arrivals[0].time))
except:
    arrival_times.append(np.nan)


# convert the times from strings to datetime object
for i in range(len(historical)):
  historical['origintime'][i] = datetime.strptime(historical['origintime'][i],'%Y-%m-%d %H:%M:%S')

# create a column to store the corrected times
historical['corrected_time'] = historical['origintime']

# fill the recently created column up with the corrected times
for i in range(len(arrival_times)):
  try:  
    historical['corrected_time'][i] = historical['corrected_time'][i] + timedelta(seconds = arrival_times[i])
  except:
    historical['corrected_time'][i] = np.nan


# we can save the data frame containing the corrected time as a csv file
# for further analysis
historical.to_csv('/Users/josea/Desktop/Big_Data/Earthquake_detection/historical_corrected.csv')

