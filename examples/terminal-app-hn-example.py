from deoxys.database import MongoDBClient
from deoxys.experiment import MultiExperimentDB
from deoxys.utils import read_file
import numpy as np
import matplotlib.pyplot as plt


def number_picker(message):
    while True:
        print('\n')
        i = input(message)
        try:
            i = int(i)
            return i
        except ValueError:
            print('\nPlease pick a number')


def action_picker(action_num, message='\nPick an option: '):
    i = -1
    while i not in np.arange(0, action_num):
        i = input(message)
        try:
            i = int(i)
        except ValueError:
            pass
    return i


def run_new_session(me, exp_id):
    tried = 0
    while tried < 3:
        try:
            epoch = number_picker('Number of epochs: ')
            path = input('Path to logs: ')
            me.run_new_session(exp_id, epochs=epoch, model_checkpoint_period=1,
                               #    prediction_checkpoint_period=1,
                               log_base_path=path)
            print('Finished running the session.')

            return True
        except Exception as e:
            print(e)
            print('Error! Please try again')
            tried += 1

    print('Failed to run session. Back to previous page')
    return True


def continue_session(me, session_id):
    tried = 0
    while tried < 3:
        try:
            epoch = number_picker('Number of epochs: ')
            path = input('Path to logs: ')
            me.continue_session(
                session_id, epoch, log_base_path=path,
                model_checkpoint_period=1,
                # prediction_checkpoint_period=1
            )
            print('Finished running the session.')

            return True
        except Exception as e:
            print(e)
            print('Error! Please try again')
            tried += 1

    print('Failed to run session. Back to previous page')
    return True


def show_performance_session(me, session_id):
    # print('Not yet supported.')
    ax = me.session_performance(session_id)
    ax.set_title('All metrics')
    plt.show()

    return True


def add_experiment(me):
    name = input('Name: ')
    description = input('Description: ')

    tried = 0
    while tried < 3:
        try:
            filepath = input('Path to config file: ')
            config = read_file(filepath)

            me.new_experiment_from_full_config(
                name, config, description=description)

            print('New experiment added successfully.')

            return True
        except Exception as e:
            print(e)
            print('Error. Try again.')
            tried += 1
    print('Failed to add experiment. Back to previous page')
    return True


def list_sessions(me, exp_id):
    sessions = me.sessions_from_experiments(exp_id)
    print(sessions)
    num_session = len(sessions)

    print('\n===============================================================')
    print('Actions: ')
    print('---------------------------------------------------------------')
    print('0. Run new session')
    if num_session > 0:
        print('1. Continue a session')
        print('2. Check performance of a session')
        print('3. Exit')
        print('-------------------------------------------'
              '--------------------')
        choice = action_picker(4)
    else:
        print('1. Exit')
        print('--------------------------------------------'
              '-------------------')
        choice = action_picker(2)

    back = False
    if choice == 0:
        while not back:
            back = run_new_session(me, exp_id)
    elif choice == 1 and num_session > 0:

        i = action_picker(num_session, '\nPick a session: ')
        session_id = me.dbclient.get_id(sessions.loc[i])

        while not back:
            back = continue_session(me, session_id)
    elif choice == 2:
        i = action_picker(num_session, '\nPick a session: ')
        session_id = me.dbclient.get_id(sessions.loc[i])

        while not back:
            back = show_performance_session(me, session_id)

    return (choice == 3 and num_session > 0) or \
        (choice == 1 and num_session == 0)


def show_performance(me, exp_id):
    # print('Not yet supported')

    ax = me.experiment_performance(exp_id)
    ax.set_title('All metrics')
    plt.show()

    return True


def list_experiment(me):
    exps = me.experiments
    print('\n===============================================================')
    print('Experiment List:')
    print('---------------------------------------------------------------')
    print(exps[['name', 'description']])
    num_exp = len(exps)

    print('\n===============================================================')
    print('Actions: ')
    print('---------------------------------------------------------------')
    print('0. Add an experiment')
    if num_exp > 0:
        print('1. View sessions of an experiment')
        print('2. Check performance of an experiment')
        print('3. Exit')
        choice = action_picker(4)
    else:
        print('1. Exit')
        choice = action_picker(2)
    print('---------------------------------------------------------------')

    back = False
    if choice == 0:
        while not back:
            back = add_experiment(me)
    elif choice == 1 and num_exp > 0:
        i = action_picker(num_exp, '\nPick an experiment: ')
        exp_id = me.dbclient.get_id(exps.loc[i])

        while not back:
            back = list_sessions(me, exp_id)
    elif choice == 2:
        i = action_picker(num_exp, '\nPick an experiment: ')
        exp_id = me.dbclient.get_id(exps.loc[i])

        while not back:
            back = show_performance(me, exp_id)

    return (choice == 3 and num_exp > 0) or (choice == 1 and num_exp == 0)


def main_options(me):
    print('===============================================================')
    print('Actions')
    print('---------------------------------------------------------------')
    print('0. List all experiements')
    print('1. Exit')
    print('---------------------------------------------------------------')

    choice = action_picker(2)

    if choice == 0:
        back = False
        while not back:
            back = list_experiment(me)

    return choice == 1


if __name__ == '__main__':
    dbclient = MongoDBClient('deoxys', 'localhost', 27017)
    me = MultiExperimentDB(dbclient)
    exps = me.experiments

    exit = False
    while not exit:
        exit = main_options(me)
    # print(exps[['name', 'description']])
