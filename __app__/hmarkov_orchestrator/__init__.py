'''
This is the Azure Durabe Function orchestration function, which
organizes the activity functions.

At two places it groups parallel tasks into a
fan-out/fan-in pattern. It also calls some activity
functions in a sequential pattern.
'''

import logging
import json
import numpy as np

import azure.functions as func          # pylint: disable=import-error
import azure.durable_functions as df    # pylint: disable=import-error

from azure.storage.blob import BlobServiceClient  # pylint: disable=all


def orchestrator_function(context: df.DurableOrchestrationContext):
    input = context.get_input()

    runs = input['runs']

    data = {'y': input['y'], 'x': input['x'], 'regimes': input['regimes'], 'maxit': input['maxit'],
            'onesigma': input['onesigma'], 'disp': input['disp'], 'convergence': input['convergence']}

    # This section runs the optimizations in parallel.
    parallel_tasks = []
    for r in range(runs):
        data['run_number'] = r
        data_str = json.dumps(data)

        parallel_tasks.append(
            context.call_activity('hmarkov_em_solve', data_str))

    parallel_outputs = yield context.task_all(parallel_tasks)

    # # This section runs sequentially, and it's intended as a comparison
    # # to test the
    # # parallelization of the previous section.
    #
    # parallel_outputs = []
    # for r in range(runs):
    #     data['run_number'] = r
    #     data_str = json.dumps(data)

    #     o = yield context.call_activity('hmarkov_em_solve', data_str)
    #     parallel_outputs.append(o)

    outputs = [json.loads(po) for po in parallel_outputs]

    llf_all = np.asarray([o['llf'] for o in outputs], dtype=np.float64)

    llf_argmax = np.argmax(llf_all)
    llf_max = llf_all[llf_argmax]

    # Note we take the maximum of the likelihoods. This is because
    # we're using an iterative routine from James D. Hamilton. If we
    # were calling an optimizer, people would usually miminize the
    # negative log likelihood.
    res = outputs[llf_argmax]
    res['llf_all'] = llf_all.tolist()
    res['llf_max'] = llf_max

    iterations_all = np.asarray([o['iterations']
                                 for o in outputs], dtype=np.float64)

    res['iterations_all'] = iterations_all.tolist()
    res['iterations'] = iterations_all[llf_argmax]

    res['time_start'] = input['start_time']

    function_times = list()
    for o in outputs:
        function_times.append(
            {'run_number': o['run_number'], 'start_time': o['start_time'], 'end_time': o['end_time'], })

    res['function_times'] = function_times

    if input['disp']:
        logs = [o['log_info'] for o in outputs]
        res['logs_all'] = logs

    # Output  paths to be saved to Azure blob.
    full_path = f"{input['account_url']}{input['base_container']}/{input['unique_id']}/"

    res['input_file'] = f'{full_path}inputs.json'
    res['output_file'] = f'{full_path}outputs.json'
    res['chart_file'] = f'{full_path}charts.png' if input['create_charts'] else None
    res['account_url'] = input['account_url']

    # Charts and t-statistics and email if optioned.
    parallel_tasks_2 = []
    if input['tstats']:
        parallel_tasks_2.append(
            context.call_activity('hmarkov_t_statistics', json.dumps(res)))

    if input['create_charts']:
        charts_str = json.dumps(
            {'res': res, 'container_name': input['base_container'], 'blob_name': input['unique_id']})

        parallel_tasks_2.append(
            context.call_activity('hmarkov_charts', charts_str))

    if len(parallel_tasks_2) > 0:
        parallel_outputs_2 = yield context.task_all(parallel_tasks_2)

    if input['tstats']:
        tstat_results = json.loads(parallel_outputs_2[0])

        res['t_statistics'] = tstat_results['t_statistics']
        res['t_statistics_time'] = tstat_results['t_statistics_time']
        res['hessian'] = tstat_results['hessian']

    # Save results to blob.
    out = json.dumps(res, indent='\t', sort_keys=True)

    save_params = json.dumps({
        'out': out, 'account_url': input['account_url'], 'container': input['base_container'], 'blob': input['outputs_file_name']})

    yield context.call_activity('hmarkov_save_outputs', save_params)

    # After everything else is done, send an email if optioned.
    if input['notification_email_address'] is not None:
        email_str = json.dumps({'input_file': res['input_file'], 'output_file': res['output_file'], 'chart_file': res['chart_file'],
                                'time_total': res['time_solve'], 'email_address': input['notification_email_address']})

    yield context.call_activity('hmarkov_email', email_str)

    # return None as a string because the Durable Function
    # extension seems to work well with strings.
    return 'None'


main = df.Orchestrator.create(orchestrator_function)
