import numpy as np
import pandas as pd
import os
from time import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

nodes = {} 
node_name = ''   # insert node name to forecast
nodeid = nodes[node_name]

# read in data as numpy arrays
X_train = np.array(#filepath)
X_val = 
X_test = 
y_train = 
y_val = 
y_test = 

# set model input shape
# first dimension is batch dimension (sample size) and is treated as None
# therefore we use every dimension after the batch dimension
# most of our models are based on tabular data and will have 1 dimension

# use following shaping for CNNs/RNNs
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

input_shape = tuple(X_train.shape[1:])

model_type = 'dense'

input = Input(shape=input_shape)
x = (Dense(500))(input)
x = (BatchNormalization())(x)
x = (ReLU())(x)
x = (Dropout(0.1))(x, training=True)
x = (Dense(500))(x)
x = (BatchNormalization())(x)
x = (ReLU())(x)
x = (Dropout(0.1))(x, training=True)
x = (Dense(500))(x)
x = (BatchNormalization())(x)
x = (ReLU())(x)
x = (Dropout(0.1))(x, training=True)
output = (Dense(1))(x)

model = Model(input, output)    # instantiate model
model.summary()

# PARAMETER DASHBOARD
# ------------------------------------------------------------------------------
# MODEL PATH

# models should be saved locally
model_dir = #ilepath
model_file = model_type + str(nodeid) + '.h5'
model_path = os.path.join(model_dir, model_file)
# ------------------------------------------------------------------------------
# HYPERPARAMETERS

# optimizer
optimizer = RMSprop(        # instantiate optimizer if not using default params
        lr=0.001,           # learning rate
        clipnorm=1.0        # limits magnitude of gradient adjustments
    )
loss = 'mse'                # loss metric to optimize
metrics = ['mae']           

# training
epochs = 500
batch_size = 64
validation_data = (X_val, y_val)
# ------------------------------------------------------------------------------
# CALLBACKS

callbacks = [
    ReduceLROnPlateau(      
        monitor='val_loss',		# performance metric to measure 
        factor=0.1,     		# scalar reduction of lr
        patience=15,    		# decrease lr if monitor doesn't improve after n epochs
        cooldown=0,     		# wait n epochs before patience countdown
        verbose=1   			# print epoch(s) where lr is reduced
    ),
    EarlyStopping(
        monitor='val_loss',		# performance metric to measure
        patience=40		        # wait n epochs before stopping training
    ),
    ModelCheckpoint(
        filepath=model_path, 	# destination to save best models
        save_best_only=True
    )
]
# ------------------------------------------------------------------------------

# compile model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics  
)

# # mlflow logging
# import mlflow
# from mlflow.tracking import MlflowClient
# from mlflow.keras import autolog
# from ABBPyFramework.Common.Utilities import createMlflowRun

# mlflow.set_tracking_uri(#hostname)
# client = MlflowClient()
# run = createMlflowRun(#run_name)

# with mlflow.start_run(experiment_id=run.info.experiment_id):
#     autolog()
#     mlflow.set_tags(
#         {
#             'model_type': model_type,
#             'node': node_name,
#             'target': 'lmp_rth',
#             'train_set': splits['train_startdate'] + ' - ' + splits['train_enddate'],
#             'validation_set': splits['val_startdate'] + ' - ' + splits['val_enddate'],
#             'scaler': 'robust',
#             'epochs': epochs,
#             'batch_size': batch_size,
#             'loss_metric': loss
#         }
#     )
    
# record training time
start_time = time()

# fit data to model
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=validation_data,
    callbacks=callbacks,
    shuffle=True
)

end_time = time()
runtime = round((end_time - start_time), 2)
print("\nTotal train time: {}".format(runtime))
    
    # mlflow.log_metrics(
    #     {
    #         'min_train_rmse': round(np.sqrt(min(history.history['loss'])), 4),
    #         'min_train_mae': round(min(history.history['mean_absolute_error']), 4),
    #         'min_val_rmse': round(np.sqrt(min(history.history['val_loss'])), 4),
    #         'min_val_mae': round(min(history.history['val_mean_absolute_error']), 4),
    #         'min_val_mape': round(min(history.history['val_mean_absolute_percentage_error']), 4)
    #     }
    # )

# MAPE (cause bret wants it)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# validation set loss
model.load_weights(model_path)
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
print("\nTrain set RMSE: {}".format(round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 3)))
print("Train set MAE: {}".format(round(mean_absolute_error(y_train, y_train_pred), 3)))
print("Val set RMSE: {}".format(round(np.sqrt(mean_squared_error(y_val, y_val_pred)), 3)))
print("Val set MAE: {}".format(round(mean_absolute_error(y_val, y_val_pred), 3)))
print("Val set MAPE: {}".format(round(mean_absolute_percentage_error(y_val, y_val_pred), 3)))

# write residuals to file
residuals = {
    'actual': y_val.flatten(),
    'pred': np.array(y_val_pred).flatten()
}
residuals = pd.DataFrame(residuals)
residuals.to_csv(
    #filepath + str(nodeid) + '.csv',
    index=False
)

# loss curve
loss = history.history['loss']
val_loss = history.history['val_loss']

train_epochs = range(1, len(loss) + 1)

plt.plot(train_epochs, loss, 'ko', label='Training loss')
plt.plot(train_epochs, val_loss, 'm', label='Validation loss')
plt.title("Training and validation loss: {}".format(node_name))
plt.legend()
plt.show()

# val set predictions vs actuals
plt.figure(figsize=(16, 6))
plt.plot(y_val, 'k', alpha=0.9, label='Val actuals')
plt.plot(y_val_pred, 'm--', alpha=0.8, label='Val predictions')
plt.title("Price forecast: {}".format(node_name))
plt.xlabel("Timesteps")
plt.ylabel("Actual price")
plt.legend(loc='upper right')
plt.show()


# # final test (run after validation is complete)
# # test set loss
# y_test_pred = model.predict(X_test)
# print("\nTest set RMSE: {}".format(round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 3)))
# print("Test set MAE: {}".format(round(mean_absolute_error(y_test, y_test_pred), 3)))
# print("Test set MAPE: {}".format(round(mean_absolute_percentage_error(y_test, y_test_pred), 3)))

# # test set predictions vs actuals
# plt.figure(figsize=(16, 6))
# plt.plot(y_test, 'k', alpha=0.9, label='Test actuals')
# plt.plot(y_test_pred, 'g--', alpha=0.8, label='Test predictions')
# plt.title("Price forecast: {}".format(node_name))
# plt.xlabel("Timesteps")
# plt.ylabel("Actual price")
# plt.legend(loc='upper right')
# plt.show()