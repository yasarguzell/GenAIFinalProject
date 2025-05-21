# GenAIFinalProject

 ## Variational Autoencoder for Synthetic Insurance Data Generation
This project implements a Variational Autoencoder (VAE) using TensorFlow and Keras to generate realistic synthetic insurance data based on the popular insurance.csv dataset. The goal is to explore how deep generative models can learn meaningful latent representations from structured tabular data that includes both numerical and categorical features.

 ## Dataset Overview:
The dataset includes the following features:

- age: Age of the individual (numerical)
- sex: Gender (categorical)
- bmi: Body Mass Index (numerical)
- children: Number of children (numerical)
- smoker: Smoking status (categorical)
- region: Residential region (categorical)
- charges: Medical insurance cost (numerical)

 ## Key Features:

- Preprocessing with ColumnTransformer:
- One-hot encoding for categorical variables
- Standard scaling for numerical variables

## VAE Architecture:

- Encoder compresses input into a latent space
- Decoder reconstructs inputs from sampled latent vectors
- Custom training loop with reconstruction loss and KL divergence

## Output:

- Synthetic data that resembles the distribution of real insurance records
- Ready for downstream tasks such as simulation, privacy-preserving analytics, or benchmarking ML models

 ## Technologies Used

- Python
- TensorFlow / Keras
- Pandas & NumPy
- scikit-learn

