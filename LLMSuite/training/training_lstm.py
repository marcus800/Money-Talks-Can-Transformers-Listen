import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from LLMSuite.evaluation.LSTM import LabelLSTM

# Assuming `features` and `target` are your inputs and outputs prepared as PyTorch tensors

data_types = ["S&P","FX"]
for data_type in data_types:

    csv_location = f"train_data/{data_type}/data.csv"
    df = pd.read_csv(csv_location)

    input_size = 1  # Number of features per time step
    hidden_size = 128  # LSTM hidden size
    num_classes = 5
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    price_change_map = {
    "MAJOR_INCREASE": 0,
    "MINOR_INCREASE": 1,
    "NO_CHANGE": 2,
    "MINOR_DECREASE": 3,
    "MAJOR_DECREASE": 4
    }
    df['label_mapped'] = df['label'].map(price_change_map)

    features = df['label_mapped'].shift(-1).fillna(method='ffill').to_frame()
    features = features.rename(columns={'label_mapped': 'shift_1'})
    features['shift_2'] = df['label_mapped'].shift(-2).fillna(method='ffill')
    features['shift_3'] = df['label_mapped'].shift(-3).fillna(method='ffill')
    target = df['label_mapped']


    def prepare_sequences(df):
        sequences = []
        targets = []
        for i in range(len(df) - 3):
            sequence = df['label_mapped'][i:i+3].values
            target = df['label_mapped'][i+3]
            sequences.append(sequence)
            targets.append(target)
        return torch.tensor(sequences, dtype=torch.float).unsqueeze(-1), torch.tensor(targets, dtype=torch.long)

    features, target = prepare_sequences(df)

    # Assuming other parts of your setup (model definition, etc.) are correct

    # Adjusting the DataLoader
    train_dataset = TensorDataset(features, target)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    #%%


    # DataLoader
    train_dataset = TensorDataset(features, target)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = LabelLSTM(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f'../saved_models/{data_type}/lstm_model.pth')