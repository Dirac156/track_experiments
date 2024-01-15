import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from wandb_scripts import WandBIntegration

# Initialize WandB Integration
wandb_integration = WandBIntegration(
    project_name="simple_model_training",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "SimpleModel"
    }
)

wandb_integration.init_run()

# Dummy Data and Model (Replace with actual data and model)
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randint(0, 2, (100,))  # Binary labels
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):  # let's assume 10 epochs
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        # Log metrics to WandB
        wandb_integration.log_metrics({"loss": loss.item()}, step=epoch * len(train_loader) + batch_idx)

# Save the model
torch.save(model.state_dict(), "simple_model.pth")
wandb_integration.save_model("simple_model.pth")

# Finish the WandB run
wandb_integration.finish_run()
