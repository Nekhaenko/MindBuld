import pandas as pd
import json
import os
import pickle
from scipy.sparse import csr_matrix
import numpy as np
import itertools
import math
from joblib import dump, load
from lightfm import LightFM
# from skopt import forest_minimize

folder = 'opt/ml/input/config/'

PATHS = {
    'hyperparameters': 'input/config/hyperparameters.json',
    'input': 'input/config/inputdataconfig.json',
    'config': 'input/config/config.json',
    'data': 'input/data/',
    'model': 'model/'
}

def read_user_data(priority):
    File_name = 'opt/ml/input/data/train/studentAnalytics2.json'
    with open(File_name) as json_file:
        data = json.load(json_file)

    user_scored_answers = dict()

    for item in data.items():
        if item[0] == 'allQuestionStats':
            lst = item[1]
    for i in lst:
        if i['questionText'] != 'Game began':
            user_scored_answers[i['questionText'].replace('R U ', '').replace('?', '')] = i['userAnswered']

    user_scored_level = {k: ('high' if v == '4' or v == '5' else 'medium' if v == '3' else 'low')
                for (k, v) in user_scored_answers.items()}

    if int(user_scored_answers['Coping']) <= 3:
        priority.remove('Coping')
        priority = ['Coping'] + priority
    elif int(user_scored_answers['Safe']) <= 3:
        priority.remove('Safe')
        priority = ['Safe'] + priority

    return priority, user_scored_level


def prepearing(config):
    # levels = config['levels']
    priority = config['priority']
    priority, user_scored_level = read_user_data(priority)
    questions = pd.read_csv('opt/ml/input/data/train/questions.csv')


    for pr in priority[:3]:
        questions_selected = questions[(questions['type']==pr)&
                                    (questions['level']==user_scored_level[pr])]
        
        questions_selected.reset_index(inplace=True,drop=True)
        break


    interactions = pd.read_csv('opt/ml/input/data/train/interactions.csv')

    user_interaction = pd.pivot_table(interactions, 
                                    index='user_id', 
                                    columns='quest_id', values='rating')

    user_interaction = user_interaction.fillna(0)
    user_interaction_csr = csr_matrix(user_interaction.values)
    user_id = list(user_interaction.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1


    item_dict ={}

    for i in questions_selected.index:
        item_dict[i] = questions_selected.loc[i,'question']

    questions_transformed = pd.get_dummies(questions_selected,
                                                    columns = ['avg_rating'])

    questions_csr = csr_matrix(questions_transformed.drop(['question',
                                                        'level',
                                                        'type',
                                                        'generated'], axis=1).values)
    return questions_csr


def train_model(user_interaction_csr):
    print('[train_model]')
    model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)

    model = model.fit(user_interaction_csr,
                epochs=100,
                num_threads=16, verbose=False)
    
    return model


def save_model(model):
    print('[save_model]')
    filename = get_path('model') + 'model'
    print(filename)
    dump(model, filename)
    
    print('Model Saved!')


def get_path(key):
    mypath = os.path.join('opt/ml/', PATHS[key])
    return mypath


def main(config):
    # levels = config['levels']
    # priority = config['priority']
    # priority, user_scored_level = read_user_data(priority)
    user_interaction_csr = prepearing(config)
    # user_interaction_csr = load_training_data(input_data_dir)
    model = train_model(user_interaction_csr)
    save_model(model)
    
    
if __name__ == "__main__":
    with open(f'{folder}config.json') as f:
        config = json.load(f)

    main(config)
