##### FEDERATED LEARNING FOR THORACIC DISEASE CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS AND DIFFERENTIAL PRIVACY ####
This repository consists of two distinct experiments which are; -

1. Federated Learning on CheXpert with EfficientNet-B3
2. Federated Learning with Differential Privacy on CheXpert

Each experiment explores a unique approach to secure and distributed training on medical imaging data, focusing on thoracic disease classification using chest X-rays from the CheXpert dataset.

    ##########  Federated Learning on CheXpert with EfficientNet-B3 ##########
Overview
This repository implements a Federated Learning (FL) framework for multi-label chest X-ray classification using the CheXpert dataset and a modified EfficientNet-B3 architecture. The system simulates decentralized training across multiple clients (hospitals or devices) that collaboratively train a global model without sharing raw data, ensuring data privacy and compliance.

1. Data Preparation
	Purpose: Load and preprocess CheXpert images and labels.
Key Logic:
Handles uncertain labels (-1) using a specified policy ("ones" or "zeroes").
Converts grayscale images to RGB.
Applies data augmentations like cropping, flipping, rotation, and normalization.
________________________________________
3. Model Architecture
 Purpose: Define the deep learning model for multi-label classification.
 Key Logic:
Uses a pretrained EfficientNet-B3 as backbone.
Adds a custom classification head with Linear → HardSwish → Dropout → Linear.
________________________________________
3. Federated Learning Setup
•	Purpose: Simulate decentralized training across multiple clients.
•	Key Logic:
o	Splits training data into 10 non-overlapping subsets (clients).
o	Each client trains the model locally.
o	Server aggregates model updates via Federated Averaging.
________________________________________
4. Training Loop
•	Purpose: Coordinate local training, aggregation, and model updates.
•	Key Logic:
o	Trains for a few epochs on selected clients per round.
o	Applies ReduceLROnPlateau scheduler.
o	Updates global model if validation loss improves.
o	Saves checkpoints after each round.
________________________________________
5. Evaluation
•	Purpose: Measure model performance on unseen test data.
•	Key Logic:
o	Computes AUROC for each of the 14 diseases.
o	Displays per-class and mean AUROC.
o	Plots ROC curves and saves as ROC.png


##########  Federated Learning With Differential Privacy ##########

1. Privacy-Aware Client Class
•	Purpose: Extend standard federated clients to support differential privacy (DP).
•	Key Logic:
o	Each client wraps its model and optimizer with Opacus’s PrivacyEngine.
o	Adds methods to log ε and δ after every round.
o	Tracks per-client training contributions and DP usage across rounds.
________________________________________
2. Differential Privacy Initialization
•	Purpose: Make each client's model training process DP-compliant.
•	Key Logic:
o	make_private_with_epsilon() ensures:
	Per-sample gradient clipping (max_grad_norm)
	Gaussian noise addition (calibrated for target ε, δ)
________________________________________
3. Federated Learning Loop (with DP)
•	Purpose: Train the global model privately across selected clients.
•	Key Logic:
o	Randomly selects a subset of clients each round.
o	Sends the global model to clients → local training with DP.
o	Collects and aggregates client updates using Federated Averaging.
________________________________________
4. Model Aggregation
•	Purpose: Combine client models into a new global model securely.
•	Key Logic:
o	Weighted averaging based on client data size.
o	Protects updates via differential privacy without leaking raw gradients.
________________________________________
5. Final Model Save
•	Purpose: Store the privacy-preserving global model.
