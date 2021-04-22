#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url,json={'Open':6.00269258e+04, 'High':6.00775508e+04, 'Low':6.00269258e+04, 'Close':6.00572305e+04, 
                            'Volume':0.00000000e+00, 'polarity':0.00000000e+00,'subjectivity':0.00000000e+00, 
                            'Compound':7.72000000e-02,'Negative':0.00000000e+00,'Neutral':8.85000000e-01, 
                            'Positive':1.15000000e-01})

print(r.json())

