from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30,activation="relu")(input_B)
hidden2 = keras.layers.Dense(30,activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A,hidden2])
main_output = keras.layers.Dense(1,name="main_output")(concat)
aux_output = keras.layers.Dense(1,name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A,input_B],outputs=[main_output,aux_output])

model.compile(loss=["mse","mse"], loss_weights=[0.9,0.1], optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:,:5], X_train[:,2:]
X_valid_A, X_valid_B = X_valid[:,:5], X_valid[:,2:]
X_test_A, X_test_B = X_test[:,:5], X_test[:,2:]

checkpoint_cb = keras.callbacks.ModelCheckpoint("./model/my_keras_model.h5",save_best_only=True)
history = model.fit([X_train_A,X_train_B],[y_train,y_train],
                    epochs=20,
                    validation_data=([X_valid_A,X_valid_B],[y_valid,y_valid]),
                    callbacks=[checkpoint_cb])
model = keras.models.load_model("./model/my_keras_model.h5")

mse_test = model.evaluate([X_test_A,X_test_B],[y_test,y_test])
print(mse_test)



