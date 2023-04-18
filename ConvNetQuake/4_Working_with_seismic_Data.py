#!/usr/bin/env python
# coding: utf-8

# # Working with seismic data
# 
# * This notebook was used to process the raw seismic data (i.e., normalization, detrending and demeaning), to split up and to save the splitted data into different directories (i.e., train/earthquake, train/noise, test/earthquake and test/noise) 

import os

filepath = '/content/sample_data/Seismic2/'

for Files in os.listdir(filepath):
  st = read('{}/{}'.format(filepath,Files))
  st.normalize(global_max=True)
  st.detrend('linear')
  st.detrend('demean')

  for i in range(len(st)):
    tr = st[i]
    start = tr.stats.starttime
    start_2 = datetime.strptime(str(start)[:-8],'%Y-%m-%dT%H:%M:%S')
    end = tr.stats.endtime
    end_2 = datetime.strptime(str(end)[:-8],'%Y-%m-%dT%H:%M:%S')
    dif = (end_2 - start_2)
    total_seconds = dif.total_seconds()
    dt = round(total_seconds/10) # number of 10s intervals within this trace

    for j in range(dt):
      if j == 0:        
        st_copy = st.copy()
        st_copy.trim(start, start + 10)
        for i in range(len(historical['origintime'])):
          if historical['origintime'][i] > start and historical['origintime'][i] < end:
            seismic_data = st_copy[0].data
            np.savetxt("/content/sample_data/train_earthquake/{}_{}.csv".format(str(start)[:-8], str(start+10)[:-8]), seismic_data, delimiter=",")            
            
          else:
            seismic_data = st_copy[0].data
            np.savetxt("/content/sample_data/train_noise/{}_{}.csv".format(str(start)[:-8], str(start+10)[:-8]), seismic_data, delimiter=",")
      else:
        st_copy = st.copy()
        st_copy.trim(start + 10*j , start + 10*(j+1))
        for i in range(len(historical['origintime'])):
          if historical['origintime'][i] > (start + 10*j) and historical['origintime'][i] < (end + 10*(j+1)):
            seismic_data = st_copy[0].data
            np.savetxt("/content/sample_data/train_earthquake/{}_{}.csv".format(str(start + 10*j)[:-8],str(start + 10*(j+1))[:-8]), seismic_data, delimiter=",")            
            
          else:
            seismic_data = st_copy[0].data
            np.savetxt("/content/sample_data/train_noise/{}_{}.csv".format(str(start + 10*j)[:-8],str(start + 10*(j+1))[:-8]), seismic_data, delimiter=",")            

