import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import _pickle as pickle
import matplotlib.pyplot as plt


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


def func_jsonParser(file, chunk_size = 1e4, chunks_num = None, save_output = False):
    
    '''
    this function reads a json file and returns a dataframe.
    '''
    
    jsonRFile = pd.read_json(file, lines = True, chunksize = chunk_size)
    df_output = pd.DataFrame()
    
    try:
        
        for line in jsonRFile:
            df_output = pd.concat([df_output, line])

            if chunks_num != None and (df_output.shape[0] / chunk_size) >= chunks_num:
                break
    
    except ValueError:
        print ("\n The file {} cannot be parsed!".format(file))
    
    #----------------------------------------------------------------------------------------#
    
    if save_output:
        with open(r"{}_chunks{}.pickle".format(file.split('.')[0], chunks_num), "wb") as fout:
            pickle.dump(df_output, fout)
        
    #----------------------------------------------------------------------------------------#
    
    return df_output


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


def func_TrainValidationTest_Splitter(list_indexes, test_percentage, training_percentage):
    
    '''
    this function returns training, validation and test indexes from the input list of indexes 
    by considering test_percentage and training_percentage.
    '''
    
    if test_percentage < 0.0 or test_percentage > 1.0:
        
        test_percentage = 0.15
        print("Test percentage is out of range, thus it was setted to {}!".format(test_percentage))
        
    if training_percentage < 0.0 or training_percentage > 1.0:
        
        training_percentage = 0.75
        print("Training percentage is out of range, thus it was setted to {}!".format(training_percentage))
    
    #-------------------------------------------------------------------#
    
    indexes_test = list_indexes[0:int(test_percentage * len(list_indexes))]
    
    cutoff1 = int(test_percentage * len(list_indexes))
    cutoff2 = int(training_percentage * ((1.0 - test_percentage) * len(list_indexes)))
    
    indexes_training = list_indexes[cutoff1:(cutoff1 + cutoff2)]
    indexes_validation = list_indexes[(cutoff1 + cutoff2):]
    
    return indexes_training, indexes_validation, indexes_test


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


def func_TrainValidationTest_Splitter_Dataframe(dataset, test_percentage, training_percentage):
    
    '''
    this function returns training, validation and test dataframes from the input dataframe
    by considering test_percentage and training_percentage.
    '''
    
    if test_percentage < 0.0 or test_percentage > 1.0:
        
        test_percentage = 0.15
        print("Test percentage is out of range, thus it was setted to {}!".format(test_percentage))
        
    if training_percentage < 0.0 or training_percentage > 1.0:
        
        training_percentage = 0.75
        print("Training percentage is out of range, thus it was setted to {}!".format(training_percentage))
    
    #-------------------------------------------------------------------#
    
    df_test = dataset.iloc[0:int(test_percentage * dataset.shape[0]), :]
    
    cutoff1 = int(test_percentage * dataset.shape[0])
    cutoff2 = int(training_percentage * ((1.0 - test_percentage) * dataset.shape[0]))
    
    df_training = dataset.iloc[cutoff1:(cutoff1 + cutoff2), :]
    df_validation = dataset.iloc[(cutoff1 + cutoff2):, :]
    
    return df_training, df_validation, df_test

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class custom_Callback_sparse(tf.keras.callbacks.Callback):
    
    '''
    this functions is used to stop the training process if a certain condition,
    for instance (sparse_categorical_accuracy > 0.95), is satisfied.
    '''
    
    def on_epoch_end(self, epoch, logs={}):
        
        if logs.get('sparse_categorical_accuracy') > 0.95:
            
            print("\n Reached 95% sparse_categorical_accuracy, so cancelling training!")
            self.model.stop_training = True


class custom_Callback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        
        if logs.get('accuracy') > 0.95:
            print("\n Reached 95% accuracy, so cancelling training!")
            self.model.stop_training = True


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


def fun_countplot(data, xlabel, ylabel, fig_width, fig_height, fig_miny, fig_maxy, 
                  fig_ticksr = 0, fig_name = '', fig_sort = True, fig_log = True):
    
    '''
    function description!
    '''
    
    fig, ax = plt.subplots(figsize = (fig_width, fig_height))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if fig_sort:
        sns.countplot(data, order = data.value_counts().index)
    else:
        sns.countplot(data)
    
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    
    plt.xticks(rotation = fig_ticksr, fontsize = 14)
    
    if fig_log:
        ax.set_yscale('log')
        plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], fontsize = 14)
    
    plt.ylim(fig_miny, fig_maxy)
    plt.show()
    
    if fig_name != '':
        fig.savefig(fig_name, dpi = 300, bbox_inches = 'tight')


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


def fun_hist(data, xlabel, ylabel, n_bins, fig_width, fig_height, fig_miny, fig_maxy, 
             fig_ticksr = 0, fig_name = ''):
    
    '''
    function description!
    '''
    
    fig, ax = plt.subplots(figsize = (fig_width, fig_height))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    
    plt.hist(data, bins = n_bins, color = np.random.rand(3,))
    
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    
    plt.xticks(rotation = fig_ticksr, fontsize = 14)
    plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], fontsize = 14)
    
    plt.xlim(0, np.max(data))
    plt.ylim(fig_miny, fig_maxy)
    
    plt.show()
    
    if fig_name != '':
        fig.savefig(fig_name, dpi = 300, bbox_inches = 'tight')


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#--------------------------------------------------------------------------------------------#
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#