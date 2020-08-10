import logging
import json

import azure.functions as func          # pylint: disable=import-error
import azure.durable_functions as df    # pylint: disable=import-error


def main(inputs: str, sendGridMessage: func.Out[str]) -> str:

    info = json.loads(inputs)

    email_content = f"input file:\n{info['input_file']}\n\noutput file:\n{info['output_file']}\n\nchart file:\n{info['chart_file']}\n"

    email_msg = {
        "personalizations": [{
            "to": [{
                "email": info['email_address']
            }]}],
        "subject": "Regime Regressions notification from Azure Functions (email sent with SendGrid)",
        "content": [{
            "type": "text/plain",
            "value": email_content}]}

    sendGridMessage.set(json.dumps(email_msg))

    # return None as a string because the Durable Function
    # extension seems to work well with strings.
    return 'None'
