from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import dump
from surprise.accuracy import rmse
from FR_JIZHI import FR_JIZHI
from FR_Weight import FR_Weight
from FS_Threshold import FS_Threshold
from FS_Weight import FS_Weight
from GFusion import GFusion
from surprise import GridSearch
import pandas as pd
from time import gmtime, strftime

reader = Reader(line_format='user item rating', sep=':')
data = Dataset.load_from_file('new_usr_ratings.txt', reader=reader)

data.split(n_folds=5)


def test_fr_jizhi():

    param_grid = {'k': [10, 20, 30], 'alpha':[0, 0.25, 0.5, 0.75, 1]}

    grid_search = GridSearch(FR_JIZHI, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)

    result_df.to_csv('FR_JIZHI_' + strftime('%m-%d-%H-%M', gmtime()))

def test_fr_weight():
    param_grid = {'k': [10, 20, 30], 'alpha': [0, 0.25, 0.5, 0.75, 1], 'beta':[0, 0.2, 0.4, 0.6, 0.8]}

    grid_search = GridSearch(FR_Weight, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)

    result_df.to_csv('FR_WEIGHT_' + strftime('%m-%d-%H-%M', gmtime()))

def test_fs_threshold():

    param_grid = {'k': [10, 20, 30], 'alpha': [0, 0.25, 0.5, 0.75, 1], 'threshold': [5, 10, 15, 20]}

    grid_search = GridSearch(FS_Threshold, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)

    result_df.to_csv('FS_threshold_' + strftime('%m-%d-%H-%M', gmtime()))

def test_fs_weight():

    param_grid = {'k': [10, 20, 30], 'alpha': [0, 0.25, 0.5, 0.75, 1], 'beta': [0, 0.2, 0.4, 0.6, 0.8]}

    grid_search = GridSearch(FS_Weight, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)

    result_df.to_csv('FS_Weight_' + strftime('%m-%d-%H-%M', gmtime()))

def test_gfusion():

    param_grid = {'k': [10, 20, 30], 'alpha': [0, 0.25, 0.5, 0.75, 1]}

    grid_search = GridSearch(GFusion, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)

    result_df.to_csv('GFusion_' + strftime('%m-%d-%H-%M', gmtime()))

def test_knn():

    param_grid = {'k': [10, 20, 30]}

    grid_search = GridSearch(KNNBasic, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)

    result_df.to_csv('KNNBasic_' + strftime('%m-%d-%H-%M', gmtime()))

test_fr_jizhi()
test_gfusion()
test_fr_weight()
test_fs_weight()
test_fs_threshold()
test_knn()




