import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from sklearn.metrics import r2_score
import joblib

# Hyperparameters
batch_size = 500
NUM_GPUS = 1  # Number of GPUs to use

# Custom Prediction Writer for Multi-GPU Support
class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        results = pl_module.all_gather(predictions)
        torch.save(results, os.path.join(self.output_dir, f"predictions_0.pt"))

# Dataset Class
class PlantTraitDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train'):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode  # 'train' or 'test'

        if self.mode == 'train':
            # For training, assume the last 6 columns are targets
            self.aux_data = self.data_frame.iloc[:, 1:-6].to_numpy(dtype=float)
            self.targets = self.data_frame.iloc[:, -6:].to_numpy(dtype=float)
        else:
            # For testing, use all columns except the first one (ID column)
            self.aux_data = self.data_frame.iloc[:, 1:].to_numpy(dtype=float)
            self.targets = None  # No targets in testing

    def __len__(self):
        return len(self.aux_data)

    def __getitem__(self, idx):
        id = self.data_frame.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, str(id) + '.jpeg')
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {
            'id': id,
            'image': image,
            'aux_data': torch.from_numpy(self.aux_data[idx, :]).float()
        }

        if self.mode == 'train':
            sample['targets'] = torch.from_numpy(self.targets[idx, :]).float()

        return sample

# Feature Extraction Module using PyTorch Lightning
class FeatureExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg_lc")
        self.model.eval()

    def forward(self, images):
        with torch.no_grad():
            features = self.model(images).squeeze().cpu().numpy()
        return features

    def predict_step(self, batch, batch_idx):
        images = batch['image']
        features = self(images)
        return features, batch['aux_data'], batch.get('targets'), batch['id']

    def extract(self, data_loader, save_path=None, output_dir="pred_path"):
        pred_path = os.path.join(output_dir, "predictions_0.pt")
        
        if os.path.exists(pred_path):
            print(f"Using cached predictions from {pred_path}")
            predictions = torch.load(pred_path)
        else:
            pred_writer = CustomWriter(output_dir=output_dir, write_interval="epoch")
            trainer = pl.Trainer(accelerator='gpu', devices=NUM_GPUS, max_epochs=1, logger=False, callbacks=[pred_writer])
            trainer.predict(self, dataloaders=data_loader, return_predictions=False)

            if os.path.exists(pred_path):
                predictions = torch.load(pred_path)
            else:
                raise RuntimeError("Predictions were not generated and saved as expected.")

        # Initialize lists to hold the extracted data
        features_list = []
        aux_data_list = []
        labels_list = []
        ids_list = []

        # Adjust unpacking based on the actual structure of `predictions`
        for prediction in predictions:
            feature_batch = prediction[0]
            aux_data_batch = prediction[1]
            label_batch = prediction[2] if len(prediction) > 2 else None
            id_batch = prediction[3] if len(prediction) > 3 else None

            features_list.append(feature_batch.cpu().numpy())
            aux_data_list.append(aux_data_batch.cpu().numpy())
            if label_batch is not None:
                labels_list.append(label_batch.cpu().numpy())
            ids_list.append(id_batch.cpu().numpy())

        # Convert lists to numpy arrays
        features = np.vstack(features_list) if features_list else np.array([])
        aux_data = np.vstack(aux_data_list) if aux_data_list else np.array([])
        ids = np.concatenate(ids_list) if ids_list else np.array([])

        labels = np.vstack(labels_list) if labels_list else None

        # Save the extracted features, if required
        if save_path:
            np.savez(save_path, features=features, aux_data=aux_data, labels=labels, ids=ids)

        # Print the size of the extracted features
        print(f"Features shape: {features.shape}")
        print(f"Auxiliary data shape: {aux_data.shape}")
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
        print(f"IDs shape: {ids.shape}")

        return features, aux_data, labels, ids

    def load_extracted_features(self, load_path):
        data = np.load(load_path, allow_pickle=True)  # Allow loading object arrays
        features = data['features']
        aux_data = data['aux_data']
        labels = data.get('labels')
        ids = data['ids']
        return features, aux_data, labels, ids

# Training with a single train/test split using XGBoost
def train_xgboost(features, aux_data, labels):
    # Check if the model already exists
    model_path = './xgboost_models/model.json'
    scaler_path = './xgboost_models/scalers.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Model already trained. Loading the existing model.")
        model = xgb.Booster()
        model.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler_features, scaler_labels = joblib.load(f)
        return model, scaler_features, scaler_labels

    # Use train_test_split instead of KFold for a single split
    train_features, val_features, train_aux, val_aux, train_labels, val_labels = train_test_split(
        features, aux_data, labels, test_size=0.1, random_state=777
    )

    print(f"Training on a single train/val split")

    # Apply StandardScaler to both features and targets
    scaler_features = StandardScaler()
    scaler_labels = StandardScaler()

    combined_train_features = np.concatenate([train_features, train_aux], axis=1)
    combined_val_features = np.concatenate([val_features, val_aux], axis=1)

    combined_train_features = scaler_features.fit_transform(combined_train_features)
    combined_val_features = scaler_features.transform(combined_val_features)

    train_labels = scaler_labels.fit_transform(train_labels)
    val_labels = scaler_labels.transform(val_labels)

    train_data = xgb.DMatrix(combined_train_features, label=train_labels)
    val_data = xgb.DMatrix(combined_val_features, label=val_labels)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': 'cuda',  # Use GPU
        'learning_rate': 0.05,
        'max_depth': 9,
        'verbosity': 1,
    }

    evals = [(train_data, 'train'), (val_data, 'val')]

    model = xgb.train(params, train_data, num_boost_round=3000, evals=evals, early_stopping_rounds=100)

    # Predict on validation set
    val_predictions = model.predict(val_data)
    val_predictions = scaler_labels.inverse_transform(val_predictions)

    # Calculate R2 score for this split
    r2 = r2_score(scaler_labels.inverse_transform(val_labels), val_predictions)
    print(f"R2 score for the train/val split: {r2:.4f}")

    # Save model and scalers
    model.save_model(model_path)
    with open(scaler_path, 'wb') as f:
        joblib.dump((scaler_features, scaler_labels), f)

    return model, scaler_features, scaler_labels

def output_predictions(test_loader, model_path, scaler_path, output_csv='test_predictions.csv'):
    # Load model and scalers
    model = xgb.Booster()
    model.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler_features, scaler_labels = joblib.load(f)

    # Extract features from the test set
    feature_extractor = FeatureExtractor()
    features, aux_data, _, ids = feature_extractor.load_extracted_features('custom_data/test_features.npz')

    # Prepare data for prediction
    combined_test_features = np.concatenate([features, aux_data], axis=1)
    combined_test_features = scaler_features.transform(combined_test_features)

    test_data = xgb.DMatrix(combined_test_features)

    predictions = model.predict(test_data)
    predictions = scaler_labels.inverse_transform(predictions)
    
    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=['X4', 'X11', 'X18', 'X26', 'X50', 'X3112'])
    df.insert(0, 'id', ids)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size expected by DINOv2
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the training data
    train_dataset = PlantTraitDataset(csv_file='data/train.csv', img_dir='data/train_images', transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Extract and save/load features
    feature_extractor = FeatureExtractor()
    train_features_path = 'custom_data/train_features.npz'

    if os.path.exists(train_features_path):
        # Load previously extracted features
        train_features, train_aux_data, train_labels, train_ids = feature_extractor.load_extracted_features(train_features_path)
    else:
        # Extract and save features
        train_features, train_aux_data, train_labels, train_ids = feature_extractor.extract(train_loader, save_path=train_features_path, output_dir="train_predictions")

    # Train the XGBoost model
    model, scaler_features, scaler_labels = train_xgboost(train_features, train_aux_data, train_labels)

    # Load and preprocess the test data
    test_dataset = PlantTraitDataset(csv_file='data/test.csv', img_dir='data/test_images', transform=transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract and save/load features for the test set
    test_features_path = 'custom_data/test_features.npz'

    if os.path.exists(test_features_path):
        # Load previously extracted features
        test_features, test_aux_data, _, test_ids = feature_extractor.load_extracted_features(test_features_path)
    else:
        # Extract and save features
        test_features, test_aux_data, _, test_ids = feature_extractor.extract(test_loader, save_path=test_features_path, output_dir="test_predictions")

    # Use the model and scalers to make predictions on the test set
    output_predictions(test_loader, './xgboost_models/model.json', './xgboost_models/scalers.pkl', output_csv='test_predictions.csv')
