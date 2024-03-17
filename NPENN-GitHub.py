import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# Read the feature matrix and transpose it
feature_matrix = pd.read_csv('data', index_col=0).T
# Read the label matrix
label_matrix = pd.read_csv('data_label.csv')

# Label encoding, 'H' is represented by 0, others by 1
label_encoder = LabelEncoder()
label_matrix['Disease'] = label_encoder.fit_transform(label_matrix['Disease'])
label_matrix['Disease'] = label_matrix['Disease'].apply(lambda x: 0 if x == 0 else 1)

# Merge feature and label data
data = feature_matrix.join(label_matrix.set_index('Sample'))

# Split the data into features and labels
X = data.drop('Disease', axis=1)
y = data['Disease']

# Initialize Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

# List to store the AUC values for each fold
auc_values = []

# Lists to store all folds' predicted results and true labels
all_predictions = []
all_true_labels = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the Gradient Boosting model
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)


    class SelfAttention(nn.Module):
        def __init__(self, hidden_size):
            super(SelfAttention, self).__init__()
            self.hidden_size = hidden_size
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(True),
                nn.Linear(64, 1)
            )

        def forward(self, encoder_outputs):
            # encoder_outputs: (batch_size, seq_len, hidden_size)
            energy = self.projection(encoder_outputs)  # (batch_size, seq_len, 1)
            weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch_size, seq_len)
            outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_size)
            return outputs, weights


    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.attention = SelfAttention(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (batch_size, seq_len, input_size)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            gru_out, _ = self.gru(x, h0)  # gru_out: (batch_size, seq_len, hidden_size)

            # Apply self-attention
            attn_out, attn_weights = self.attention(gru_out)  # attn_out: (batch_size, hidden_size)

            # Final fully connected layer
            out = self.fc(attn_out)  # out: (batch_size, output_size)
            out = self.sigmoid(out)
            return out

    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 64
    output_size = 1
    gaussian_noise = 0.1
    uniform_noise = 0.1
    model = GRUModel(input_size, hidden_size, num_layers, output_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the GRU model
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # Concatenate Gradient Boosting output with input
        gb_predictions = gb_model.predict_proba(X_train)[:, 1]
        gb_predictions_tensor = torch.tensor(gb_predictions, dtype=torch.float32).view(-1, 1, 1)
        gru_input = torch.cat((torch.tensor(X_train.values, dtype=torch.float32).view(-1, 1, X_train.shape[1]), gb_predictions_tensor), dim=2)
        outputs = model(gru_input, gaussian_noise, uniform_noise)
        loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        gb_test_predictions = gb_model.predict_proba(X_test)[:, 1]
        gb_test_predictions_tensor = torch.tensor(gb_test_predictions, dtype=torch.float32).view(-1, 1, 1)
        gru_input = torch.cat((torch.tensor(X_test.values, dtype=torch.float32).view(-1, 1, X_test.shape[1]), gb_test_predictions_tensor), dim=2)
        y_pred = model(gru_input, gaussian_noise, uniform_noise)
        auc_value = roc_auc_score(y_test, y_pred.numpy())
        auc_values.append(auc_value)

        # Collect predictions and true labels for this fold
        all_predictions.extend(y_pred.numpy().squeeze().tolist())
        all_true_labels.extend(y_test.tolist())

# Calculate the mean AUC value
mean_auc = np.mean(auc_values)
print(f"Mean AUC: {mean_auc:.3f}")

# Save all folds' predicted results and true labels to a CSV file
results_df = pd.DataFrame({'True_Labels': all_true_labels, 'Predicted_Probability': all_predictions})
results_df.to_csv('predicted_results.csv', index=False)
print("Predicted results saved to 'predicted_results.csv'")