import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Load data
df = pd.read_csv("insurance.csv")

# 2. Separate features
cat_cols = ['sex', 'smoker', 'region']
num_cols = ['age', 'bmi', 'children', 'charges']

# 3. Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
    ("num", StandardScaler(), num_cols)
])
X = preprocessor.fit_transform(df)

input_dim = X.shape[1]
latent_dim = 5

# 4. ENCODER layer
inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(32, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(h)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

# 5. DECODER layer
latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(32, activation='relu')(latent_inputs)
outputs = layers.Dense(input_dim, activation='sigmoid')(x)

decoder = Model(latent_inputs, outputs, name="decoder")

# 6. VAE model definition
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction)) * input_dim
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')

# 7. Train the model
vae.fit(X, epochs=50, batch_size=32)

# 8. Generate new samples
z_sample = np.random.normal(size=(10, latent_dim))
generated = decoder.predict(z_sample)

# 9. Inverse transform numerical features
scaler = preprocessor.named_transformers_['num']
generated_num = generated[:, -len(num_cols):]  # Numeric features are at the end
generated_num_real = scaler.inverse_transform(generated_num)
gen_num_df = pd.DataFrame(generated_num_real, columns=num_cols)

# 10. Post-process numeric values
gen_num_df['age'] = gen_num_df['age'].round().astype(int)
gen_num_df['bmi'] = gen_num_df['bmi'].round().astype(int)
gen_num_df['children'] = gen_num_df['children'].round().astype(int)
gen_num_df['charges'] = gen_num_df['charges'].round(4)

# 11. Decode categorical one-hot vectors
encoder_cat = preprocessor.named_transformers_['cat']
cat_feature_names = encoder_cat.get_feature_names_out(cat_cols)

# 12. Calculate lengths of each categorical one-hot group
cat_feature_lengths = []
for col in cat_cols:
    length = sum([1 for name in cat_feature_names if name.startswith(col + '_')])
    cat_feature_lengths.append(length)

generated_cat = generated[:, :len(cat_feature_names)]  # Categorical part

idx = 0
decoded_cats = {}
for i, col in enumerate(cat_cols):
    length = cat_feature_lengths[i]
    cat_slice = generated_cat[:, idx:idx+length]
    idx += length
    cat_labels_idx = np.argmax(cat_slice, axis=1)
    categories = encoder_cat.categories_[i]
    decoded_cats[col] = categories[cat_labels_idx]

gen_cat_df = pd.DataFrame(decoded_cats)

# 13. Combine numeric and categorical columns
final_df = pd.concat([gen_cat_df, gen_num_df], axis=1)
print(final_df)
