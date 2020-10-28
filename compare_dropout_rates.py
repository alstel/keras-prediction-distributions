import pandas as pd
import numpy as np
import os
from time import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# specify hub before running
nodes = {}
node_name = ''
nodeid = nodes[node_name]

# load data
data_path = #filepath
X_test_path = os.path.join(data_path, 'Xtest_' + str(nodeid) + '.csv')
y_test_path = os.path.join(data_path, 'ytest_' + str(nodeid) + '.csv')
X_test = np.array(pd.read_csv(X_test_path))
y_test = np.array(pd.read_csv(y_test_path))

# load model
model_path = #filepath + str(nodeid) + '.h5'
model = load_model(model_path)

def create_dropout_predict_function(model, dropout):
    """
    Hard-codes a dropout rate to the Dropout layers of a trained model.
    When initially training model, Dropout layers must have training=True, 
    otherwise dropout will not be applied to predictions.

    Parameters:
        model : trained keras model
        dropout : dropout rate to apply to all Dropout layers
    
    Returns:
        predict_with_dropout : keras model that will apply dropout when making predictions
    """
    
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"]=="Dropout":
            layer["config"]["rate"] = dropout
        # Recurrent layers with dropout
        elif "dropout" in layer["config"].keys():
            layer["config"]["dropout"] = dropout

    # Create a new model with specified dropout
    if type(model)==Sequential:
        # Sequential
        model_dropout = Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights())
    
    return model_dropout

def dropout_comparison(model, X, y, dropout_rate, pi, num_iter):
    """
    Creates a dataframe of dropout rate comparisons.
    """
    num_samples = X.shape[0]
    # dataframe for storing val percentages
    vals_in_df = pd.DataFrame(
        columns=[
            'dropout_rate',
            'prediction_interval',
            'actuals_in_interval'
        ]
    )
    # loop through dropout rate and PI combinations
    for rate in dropout_rate:
        for interval in pi:
            # generate prediction distributions
            predict_with_dropout = create_dropout_predict_function(model, rate)
            # matrix for storing distributions
            predictions = np.zeros((num_samples, num_iter))
            for i in range(num_iter):
                predictions[:,i] = predict_with_dropout.predict(X).reshape(-1)
            # calculate upper and lower lims of PI
            lower_lim = np.quantile(predictions, 0.5-interval/2, axis=1)
            upper_lim = np.quantile(predictions, 0.5+interval/2, axis=1)
            # create DF of lims vs actuals
            intervals = pd.DataFrame(
                data={
                    'lower_lim': lower_lim,
                    'upper_lim': upper_lim,
                    'actual': y.reshape(-1) # set to 1-dimensional
                }
            )
            # set flag for actual values inside PI
            intervals['in_interval'] = intervals.apply(
                lambda row: 1
                if row['lower_lim'] <= row['actual'] <= row['upper_lim']
                else 0,
                axis=1
            )
            # percentage of actuals within PI
            vals_inside = intervals.loc[
                intervals.in_interval == 1,
                'in_interval'
            ].count()
            actuals_in_interval = round((vals_inside / len(intervals)), 3)
            # append to val percentages df
            vals_in_df = vals_in_df.append(
                {
                    'dropout_rate': rate,
                    'prediction_interval': interval,
                    'actuals_in_interval': actuals_in_interval
                },
                ignore_index=True
            )
    
    return vals_in_df

# create prediction distributions
print("\nRunning predictions...")
start = time()
num_iter = 2000
perc = dropout_comparison(
    model,
    X_test,
    y_test,
    dropout_rate=[
        0.20,
        0.40,
        0.60,
        0.80
    ],
    pi=[
        0.60,
        0.80,
        0.95
    ],
    num_iter=num_iter
)
end = time()
run_time = round((end - start), 2)
print("\nPredictions processing time: {}".format(run_time))

# summary
model.summary()
print("\n{}, {} iterations".format(node_name, num_iter))
print(perc)