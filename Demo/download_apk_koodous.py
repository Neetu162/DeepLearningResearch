#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:56:43 2020

@author: osboxes
"""


import requests
import urllib.request

sha256 = "2ec2d423433901faf9e983d8c8db5397122a0811063a62f6bbbbc142fc60c490"

url_koodous = "https://api.koodous.com/apks/%s/download" % sha256
r = requests.get(url=url_koodous, headers={'Authorization': 'Token 35682e08e89ac03e581e25fbd8c8b0122acb11eb'})
print (str(r.status_code))
if r.status_code == 200:
    urllib.request.urlretrieve(r.json().get('download_url'), "temp.apk")