import logging
import json
import os
from datetime import datetime

import azure.functions as func          # pylint: disable=import-error
import azure.durable_functions as df    # pylint: disable=import-error

from azure.storage.blob import BlobServiceClient  # pylint: disable=all


def main(inputs: str) -> str:
    inputs_json = json.loads(inputs)

    # record end time
    now = datetime.now()
    end_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    out = json.loads(inputs_json['out'])

    out['time_end'] = end_time
    out_str = json.dumps(out, indent='\t', sort_keys=True)

    # save output
    account_url = inputs_json['account_url']
    container = inputs_json['container']
    blob = inputs_json['blob']

    # Save output to Azure blob.
    blob_service = BlobServiceClient(account_url=account_url,
                                     credential=os.environ['BlobCredentials'])

    blob = blob_service.get_blob_client(container=container, blob=blob)

    blob.upload_blob(out_str)

    # return None as a string because the Durable Function
    # extension seems to work well with strings.
    return 'None'
