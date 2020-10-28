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

# for indexing later
splits = {
    'test_startdate': '2020-01-01',
    'test_enddate': '2020-01-11'
}

# load data
data_path = #filepath
X_test_path = os.path.join(data_path, 'Xtest_' + str(nodeid) + '.csv')
y_test_path = os.path.join(data_path, 'ytest_' + str(nodeid) + '.csv')
X_test = np.array(pd.read_csv(X_test_path))
y_test = np.array(pd.read_csv(y_test_path))

# load model
model_type = 'dense'
model_path = #filepath + str(nodeid) + '.h5'
model = load_model(model_path)

# thanks to sfblake on github for original function
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

# create prediction distribution
dropout = 0.4
num_samples = X_test.shape[0]
num_iter = 2000

predict_with_dropout = create_dropout_predict_function(model, dropout)

print("\nRunning predictions...")
start = time()
predictions = np.zeros((num_samples, num_iter))
for i in range(num_iter):
    predictions[:,i] = predict_with_dropout.predict(X_test).reshape(-1)
end = time()
run_time = round((end - start), 2)
print("\nPrediction distribution process time: {}".format(run_time))

# set prediction intervals and colors
prediction_intervals = {
    0.95: '#fb8072',
    0.80: '#bebada',
    0.60: '#ffffb3'
}

# plot prediction distibution
plt.figure(figsize=(23, 10))

for interval, color in prediction_intervals.items():
    lower_lim = np.quantile(predictions, 0.5-interval/2, axis=1)
    upper_lim = np.quantile(predictions, 0.5+interval/2, axis=1)
    plt.plot(lower_lim, color, alpha=1, label=str(int(interval * 100)) + '% prediction interval')
    plt.plot(upper_lim, color, alpha=1)
    plt.fill_between(range(len(predictions)), lower_lim, upper_lim, color=color, alpha=1)

#plt.plot(predictions, 'g', label="preds")
plt.plot(y_test, 'k', alpha=0.8, label="RTH")
plt.title("Price forecast: {}\n{}% dropout, {} iterations".format(
        node_name,
        str(dropout * 100),
        num_iter
    ), fontsize=16)
plt.xticks(fontsize=12)
plt.xlabel("timesteps", fontsize=14)
plt.yticks(fontsize=12)
plt.ylabel("price", fontsize=14)
plt.legend(fontsize=13)
sns.despine()
plt.show()

# write data to file
date_index = pd.date_range(
    start=splits['test_startdate'],
    end=splits['test_enddate'],
    freq='H',
    closed='left'
)
predictions_df = pd.DataFrame(data=predictions)
rth_df = pd.DataFrame(data=y_test, columns=['RTH'])
distribution_df = pd.concat(
    [
        predictions_df,
        rth_df
    ],
    axis=1
)
distribution_df.index = date_index
distribution_df.to_csv(#filepath)