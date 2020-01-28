"""
Example file of running and continuation of an experiement multiple times
"""

from deoxys.database import MongoDBClient
from deoxys.experiment import MultiExperimentDB
import numpy as np

if __name__ == '__main__':
    dbclient = MongoDBClient('deoxys', 'localhost', 27017)
    me = MultiExperimentDB(dbclient)
    exps = me.experiments

    # List experiements
    print(exps[['name', 'description']])

    # Pick one experiment
    i = -1
    while i not in np.arange(0, len(exps)):
        i = input('Pick an experiment: ')
        try:
            i = int(i)
        except ValueError:
            pass

    experiment_id = me.dbclient.get_id(exps.loc[i])

    # List all sessions
    sessions = me.sessions_from_experiments(experiment_id)
    print(sessions)

    # Start a new session
    new_session = me.run_new_session(experiment_id, epochs=4,
                                     log_base_path='../../oxford_perf/logs_db')

    # List all sessions to see the changes
    sessions = me.sessions_from_experiments(experiment_id)
    print(sessions)

    # Continue an existing session
    session_id = me.dbclient.get_id(sessions.loc[len(sessions)])
    me.continue_session(
        session_id, 2, log_base_path='../../oxford_perf/logs_db')

    # List all sessions to see the changes
    sessions = me.sessions_from_experiments(experiment_id)
    print(sessions)

    # Start multiple sessions
    me.run_multiple_new_session(2, experiment_id, 2,
                                log_base_path='../../oxford_perf/logs_db')

    # List all sessions to see the changes
    sessions = me.sessions_from_experiments(experiment_id)
    print(sessions)

    # Continue multiple sessions
    session_ids = [me.dbclient.get_id(sessions.loc[i])
                   for i in range(len(sessions) - 3, len(sessions))]
    me.continue_multiple_session(
        session_ids, 2, log_base_path='../../oxford_perf/logs_db')

    # List all sessions to see the changes
    sessions = me.sessions_from_experiments(experiment_id)
    print(sessions)
