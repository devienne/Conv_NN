#!/usr/bin/env python
# coding: utf-8

# # Download historical earthquake list
# 
# * This notebook were used to download the list of historical earthquake


import csv
import requests

URL = 'https://ogsweb.ou.edu/api/earthquake?start=201402140000&end=201611162359&mag=0&format=csv'
RESPONSE = requests.get(URL)

EXPORT = '/Users/josea/Desktop/Big_Data/Earthquake_detection/historical.csv'

with open(EXPORT, 'w') as f:
    WRITER = csv.writer(f)
    for line in RESPONSE.iter_lines():
        WRITER.writerow(line.decode('utf-8').split(','))





