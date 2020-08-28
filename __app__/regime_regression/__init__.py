'''
This is the client function, which is the entry point for
this Azure Durable Function.

It receives an HTTP request containing data and
then initiates the orchestration.

There is some data validation here. This is to give the user an
immediate response before the orchestration starts: either
feedback on a data error or a series of URLs where
the results are expected after the analysis finishes.
'''

import logging
import json
import uuid

import os
import numpy as np
from datetime import datetime, timedelta

import azure.functions as func          # pylint: disable=import-error
import azure.durable_functions as df    # pylint: disable=import-error
from azure.durable_functions.models.OrchestrationRuntimeStatus import OrchestrationRuntimeStatus  # pylint: disable=import-error

from azure.storage.blob import BlobServiceClient  # pylint: disable=all

from aiohttp import ContentTypeError


async def main(req: func.HttpRequest, starter: str) -> func.HttpResponse:

    # record start time
    now = datetime.now()
    start_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Before attempting to parse data, set up blob storage for output (including errors).
    unique_id = str(uuid.uuid4()).replace('-', '')

    account_url = 'https://regimeregressions.blob.core.windows.net/'
    base_container = 'results'

    outputs_file_name = f'{unique_id}/outputs.json'
    inputs_file_name = f'{unique_id}/inputs.json'

    # Set up credentials to save input blob (or output blob if error).
    blob_service = BlobServiceClient(account_url=account_url,
                                     credential=os.environ['BlobCredentials'])
    mimetext = 'text/plain'

    # Inner function to save input blob (or output blob if error).
    def blob_save_output(out_str, is_err=False):
        if is_err:
            err_dict = {"error_message:": out_str}
            out_str = json.dumps(err_dict, indent='\t')
            fname = outputs_file_name
        else:
            fname = inputs_file_name

        blob = blob_service.get_blob_client(
            container=base_container, blob=fname)

        blob.upload_blob(out_str)

    # GET REQUEST BODY
    try:
        info = req.get_json()
    except ValueError as ve:
        err = f"ERROR:: hmarkov function: Retrieving text from request body returned an error.\n{str(ve)}"
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    # Check for purge command. If present, purge without analysis.
    purge_past_instances = info.get('purge_past_instances', False)

    if purge_past_instances:
        keep_days = info.get('keep_days', 3)

        client = df.DurableOrchestrationClient(starter)

        created_time_from = datetime.today() + timedelta(days=-30000)
        created_time_to = datetime.today() + timedelta(days=-keep_days)
        runtime_statuses = [OrchestrationRuntimeStatus.Completed,
                            OrchestrationRuntimeStatus.Terminated, OrchestrationRuntimeStatus.Failed]

        try:
            purge_coroutine = client.purge_instance_history_by(
                created_time_from, created_time_to, runtime_statuses)

            purge_history = await purge_coroutine

        except ContentTypeError as e:
            purge_err = f'\nError: failed to delete past instances. The ContentTypeError can occur when running locally or when there are no instances to purge.\n{repr(e)}\n'

            return func.HttpResponse(purge_err, status_code=500, mimetype=mimetext)

        except Exception as e:
            purge_err = f'\nError: failed to delete past instances. Unknown error:\n{repr(e)}\n'

            return func.HttpResponse(purge_err, status_code=500, mimetype=mimetext)

        else:
            purge_msg = f'Deleted past instances, without analyzing any submitted data. Purged {purge_history.instances_deleted} past instances that were created from {created_time_from} to {created_time_to}.'

            return func.HttpResponse(purge_msg, status_code=200, mimetype=mimetext)

    # PARSE REQUEST BODY
    #
    # Set max values--Azure functions can run for maximum of 10 minutes.
    max_obs = 10000
    max_regimes = 4
    max_runs = 5

    regimes = info.get('regimes', 2)
    onesigma = info.get('onesigma', True)
    maxit = info.get('maxit', 1000)
    convergence = info.get('convergence', 1e-8)
    runs = info.get('runs', 3)
    disp = info.get('disp', False)
    create_charts = info.get('create_charts', True)
    tstats = info.get('calc_tstats', False)

    notification_email_address = info.get('notification_email_address', None)

    charts_file_name = f'{unique_id}/charts.json' if create_charts else None

    # Error-checking on inputs.
    if regimes < 1 or regimes > max_regimes:
        err = f"ERROR:: hmarkov function: {regimes} regimes input, but number of regimes must be between 1 and {max_regimes}."
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    if runs < 1 or runs > max_runs:
        err = f"ERROR:: hmarkov function: {runs} runs input, but number of parallel runs must be between 1 and {max_runs}."
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    if 'y' in info:
        try:
            y = np.asarray(info['y'], dtype=np.float64)
        except Exception as e:
            err = f"ERROR:: hmarkov function: The 'y' input array has a value that can't be converted to a number. Python exception: {e}"
            blob_save_output(err, is_err=True)
            return func.HttpResponse(err, status_code=400, mimetype=mimetext)

        if np.isnan(np.sum(y)):
            err = f"ERROR:: hmarkov function: The 'y' input has a value that can't be converted to a number."
            blob_save_output(err, is_err=True)
            return func.HttpResponse(err, status_code=400, mimetype=mimetext)
    else:
        err = f"ERROR:: hmarkov function: no data labeled 'y' in input JSON."
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    nobs = max(y.shape)
    if nobs > max_obs:
        err = f"ERROR:: hmarkov function: {nobs} observations input, but maximum is {max_obs}."
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    if 'x' in info:
        x = list()
        x_in = info['x']
        for count, xi in enumerate(x_in):
            if xi is None:
                x.append(None)
            else:
                try:
                    x.append(np.asarray(xi, dtype=np.float64))
                except Exception as e:
                    err = f"ERROR:: hmarkov function: The 'x' input array {count+1} of {len(x_in)} has a value that can't be converted to a number. Python exception: {e}"
                    blob_save_output(err, is_err=True)
                    return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    else:
        err = f"ERROR:: hmarkov function: no data labeled 'x' in input JSON. An empty 'x' needs to be labeled 'null' (for each regime) in JSON."
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    for count, xi in enumerate(x):
        if xi is not None:
            nobs_x = max(xi.shape)
            if nobs_x != nobs:
                err = f"ERROR:: hmarkov function: one of the 'x' inputs has {nobs_x} observations while the 'y' input has {nobs} observations."
                blob_save_output(err, is_err=True)
                return func.HttpResponse(err, status_code=400, mimetype=mimetext)

            if np.isnan(np.sum(xi)):
                err = f"ERROR:: hmarkov function: The 'x' input array {count+1} of {len(x)} has a value that can't be converted to a number."
                blob_save_output(err, is_err=True)
                return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    if len(x) != regimes:
        err = f"ERROR:: hmarkov function: there were {len(x)} 'x' inputs but {regimes} regimes."
        blob_save_output(err, is_err=True)
        return func.HttpResponse(err, status_code=400, mimetype=mimetext)

    # SAVE VALIDATED DATA TO BLOB
    data = {'convergence': convergence,
            'create_charts': create_charts,
            'disp': disp,
            'maxit': maxit,
            'nobs': nobs,
            'notification_email_address': notification_email_address,
            'onesigma': onesigma,
            'regimes': regimes,
            'runs': runs,
            'tstats': tstats,
            'unique_id': unique_id,
            'outputs_file_name': outputs_file_name,
            'inputs_file_name': inputs_file_name,
            'charts_file_name': charts_file_name,
            'account_url': account_url,
            'base_container': base_container,
            'y': y.tolist(),
            'start_time': start_time}

    x_list = []
    for x_item in x:
        if x_item is None:
            x_list.append(None)
        else:
            x_list.append(x_item.tolist())

    data['x'] = x_list

    blob_save_output(json.dumps(data, indent='\t', sort_keys=True))

    # Start orchestration of Azure Durable Function.
    client = df.DurableOrchestrationClient(starter)
    instance_id = await client.start_new(req.route_params["functionName"], None, data)

    logging.info(f"Started orchestration with ID = '{instance_id}'.")

    # If data passed validation, send file locations back to user through HTTP response.
    url_base = f'{account_url}{base_container}/{unique_id}/'
    http_msg = {'inputs_url': f'{url_base}inputs.json',
                'outputs_url': f'{url_base}outputs.json'
                }

    if create_charts:
        http_msg['chart_url'] = f'{url_base}chart.png'

    http_out = json.dumps(http_msg, indent='\t')

    return func.HttpResponse(http_out, status_code=200, mimetype='application/json')
