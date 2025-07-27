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

________<img width="1431" height="827" alt="Figure  2  Proposed model and FL framework" src="https://github.com/user-attachments/assets/73804ab3-8e00-42dc-b00e-0f640fed8f14" />
________________________________
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
<img width="1378" height="690" alt="Figure  4 (complete) ROC of Proposed model (Efficient net B3)" src="https://github.com/user-attachments/assets/d4fe1448-1f72-485f-a295-28e04c3f70a8" />


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

![Figure  7  Class Wise Characteristic Curves ε= 15, σ =0 3575 (Indicates Moderate Privacy)](https://github.com/user-attachments/assets/2e7b1c56-9376-49dc-9ff1-b374aa052fa9)
<img width="1172" height="834" alt="Figure  6  AUC Wise Characteristic Curves ε= 15, σ =0 3575 (Indicates Moderate Privacy)" src="https://github.com/user-attachments/assets/bc64bdc9-40bb-49b5-a15f-8c55e586afe4" />
<img width="1203" height="856" alt="Figure  8  Class Wise Characteristic Curves ε= 30, σ =0 2870 (Indicates Low Privacy)" src="https://github.com/user-attachments/assets/30945e32-39d6-4ffd-a5a7-ef8f9d1edb0c" />
