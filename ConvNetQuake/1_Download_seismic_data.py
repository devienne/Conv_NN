#!/usr/bin/env python
# coding: utf-8

# # Download seismic data
# 
# * This scripts were used to download the raw seismic data from the IRIS API

import requests # for requesting the seismic data from IRIS API 
import datetime # to deal with date in python
import pandas as pd # to manipulate dataframes


START = datetime.datetime(2014, 2, 14)
END = datetime.datetime(2014, 10, 14)

daterange = pd.date_range(START,END)


STAT = 'OK029'
CHAN = 'HHE,HHN,HHZ'
FORMAT = 'miniseed'

def download():
    for i in daterange:
        YEAR = i.date().year
        MONTH = i.date().month
        if MONTH < 10:
            MONTH = '0{}'.format(MONTH)
        DAY = i.date().day
        if DAY < 10:
            DAY = '0{}'.format(DAY)

        URL = 'http://service.iris.edu/fdsnws/dataselect/1/query?sta={}&cha={}&starttime={}-{}-{}T00:00:00&endtime={}-{}-{}T23:59:59&format={}&nodata=404'.format(STAT,CHAN,YEAR,MONTH,DAY,YEAR,MONTH,DAY,FORMAT)

        FILENAME = STAT + '_' + str(YEAR) + '-' + str(MONTH) + '-' + str(DAY)
        RESPONSE = requests.get(URL, allow_redirects = True)
        if RESPONSE.text[0:5] == 'Error':
            pass
        else:
            FILEPATH = '/Users/josea/Desktop/Big_Data/Earthquake_detection/Seismic_data/'
            FILE = FILEPATH + FILENAME + '.mseed' 
            open(FILE, 'wb').write(RESPONSE.content)


download()

