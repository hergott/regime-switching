'''
This is a demonstration of the regime-switching example written by Matt Hergott.

It creates a time series with three regimes, and the regimes are determined
by a sine wave.

It performs three regressions on the regimes to establish a baseline result.

It then calls the Azure Durable Function through Azure API Management, and those
results from the regime-switching model can be compared to the linear regressions.
'''

import numpy as np
import statsmodels.api as sm

import json
import requests

import math

def main():

    # Create time series with 3 regimes based on sine wave.
    max_rad = math.pi*10.
    n_obs = 10000
    rng = np.linspace(0, max_rad, num=n_obs, dtype=np.float64)
    sine = np.sin(rng)

    y = np.zeros((n_obs,))

    x = np.random.random_sample((n_obs,))-0.5
    
    # Add a bit of noise to make the problem a little
    # harder (i.e., stress-testing the regime-switching model).
    noise = (np.random.random_sample((n_obs,))-0.5) * 0.33

    coeffs = (0.2, 0.8, 1.4)

    regime_idx = list()

    regime_idx.append(np.where(sine < -0.33))
    regime_idx.append(np.where((sine >= -0.33) & (sine < 0.33)))
    regime_idx.append(np.where(sine >= 0.33))

    for i, r in enumerate(regime_idx):
        y[r] = coeffs[i]*x[r]+noise[r]

    # Calculate regressions for individual regimes.
    for i, r in enumerate(regime_idx):
        y_regime = y[r].copy()
        x_regime = x[r].copy()
        x_regime_intc = sm.add_constant(x_regime)

        regr = sm.OLS(y_regime, x_regime_intc)
        regr_fit = regr.fit()

        print(f'\n\nRegime {i}: \n')
        print(regr_fit.summary())


    # Now, set up the data for HTTP reqest through Azure API Management.
    calc_tstats = True
    create_charts = True
    disp = False
    
    y = y.tolist()
    x = x.tolist()

    r = {'regimes': 3, 'runs': 5, 'y': y,  'convergence': 1e-8,
         'create_charts': create_charts, 'calc_tstats': calc_tstats, disp: disp, 'x': [x, x, x]}

    # Comment out this line if you don't want an email.
    r['notification_email_address'] = '[ENTER EMAIL ADDRESS HERE FOR NOTIFICATION]'

    r_json = json.dumps(r)  

    # Call Azure function through Azure API Management.
    API_URL = 'https://mbai.azure-api.net/regime-regressions/orchestrators/hmarkov_orchestrator'
    API_KEY_HEADER_NAME = 'Ocp-Apim-Subscription-Key'
    API_KEY = '[ENTER API KEY HERE]'

    headers = {API_KEY_HEADER_NAME: API_KEY}

    req_raw = requests.post(f'{API_URL}', data=r_json, headers=headers)
    
    if req_raw.status_code != 200:
        print(req_raw.status_code)
        print(req_raw.text)
        return

    # Print the intial response. If it passed data validation,
    # this will show the URLs where the results will be.
    req = req_raw.json()
    print(req)   

    # This is an optional section that polls the results endpoint for the
    # JSON file of results. You can also just wait for the notification
    # email and then download the results.
    status_code = 404

    while status_code == 404:
        time.sleep(10)
        response = requests.get(req["outputs_url"])
        status_code = response.status_code

    print(response)
    outputs = response.json()

    print(f'betas: {outputs["beta"]}')

    if calc_tstats:
        print(f't stats: {outputs["t_statistics"]}')
        
        
main()