import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

class CNN:
    def __init__(
        self,
        input_shape=(25, 20, 1),
        branch_kernel_heights=(1, 2),
        n_blocks_per_branch=4,
        base_filters=32,
        pool_size=(2, 2),
        dense_units=256,
        dropout_rate=0.4,
        learning_rate=1e-5,
        metrics=("accuracy", tf.keras.metrics.AUC(name="auc"))
    ):
        self.input_shape = input_shape
        self.branch_kernel_heights = branch_kernel_heights
        self.n_blocks_per_branch = n_blocks_per_branch
        self.base_filters = base_filters
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.metrics = metrics

        self.model = self._build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=list(self.metrics)
        )

    def _conv_block(self, x, filters, kernel_size, pool_size, name_prefix):
        x = layers.Conv2D(filters, kernel_size, padding="same",
                          kernel_initializer=GlorotUniform(),
                          name=f"{name_prefix}_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=f"{name_prefix}_pool")(x)
        return x

    def _build_branch(self, inputs, kernel_height, branch_name):
        x = inputs
        filters = self.base_filters
        for i in range(self.n_blocks_per_branch):
            x = self._conv_block(
                x,
                filters=filters,
                kernel_size=(kernel_height, 3),
                pool_size=self.pool_size,
                name_prefix=f"{branch_name}_block{i+1}"
            )
            filters = min(512, filters * 2)

        x = layers.Conv2D(filters, (1, 1), padding="same",
                          kernel_initializer=GlorotUniform(),
                          name=f"{branch_name}_final_conv")(x)
        x = layers.BatchNormalization(name=f"{branch_name}_final_bn")(x)
        x = layers.Activation("relu", name=f"{branch_name}_final_relu")(x)
        x = layers.Flatten(name=f"{branch_name}_flatten")(x)
        return x

    def _build_model(self):
        inputs = Input(shape=self.input_shape, name="input_layer")

        branch1 = self._build_branch(inputs, self.branch_kernel_heights[0], "branch1")
        branch2 = self._build_branch(inputs, self.branch_kernel_heights[1], "branch2")

        merged = layers.Concatenate(name="concat")([branch1, branch2])

        x = layers.Dense(self.dense_units, activation="relu", kernel_initializer=GlorotUniform(), name="fc1")(merged)
        x = layers.BatchNormalization(name="fc1_bn")(x)
        x = layers.Dropout(self.dropout_rate, name="fc1_dropout")(x)

        x = layers.Dense(self.dense_units // 2, activation="relu", kernel_initializer=GlorotUniform(), name="fc2")(x)
        x = layers.BatchNormalization(name="fc2_bn")(x)
        x = layers.Dropout(self.dropout_rate, name="fc2_dropout")(x)

        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs, outputs, name="ParallelCNN")
        return model

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        early_stopping=False,
        patience=10,
        verbose=1,
        model_dir="results"
    ):
        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, "best_model.h5")

        callbacks = [
            ModelCheckpoint(
                best_model_path,
                monitor="val_accuracy",  # ✅ track validation accuracy
                mode="max",
                save_best_only=True,
                verbose=1
            )
        ]

        if early_stopping and X_val is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        # --- Retrieve best epoch ---
        best_epoch = np.argmax(history.history["val_accuracy"]) + 1
        best_val_acc = max(history.history["val_accuracy"])

        # Load best weights
        self.model.load_weights(best_model_path)

        print(f"\n✅ Best Epoch: {best_epoch} | Best Validation Accuracy: {best_val_acc:.4f}")

        return {
            "history": history.history,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_model_path": best_model_path
        }

    def evaluate(self, X_test, y_test, batch_size=32):
        return self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

    def predict(self, X, batch_size=32):
        return self.model.predict(X, batch_size=batch_size, verbose=1)

    def summary(self):
        self.model.summary()

    def save(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)
