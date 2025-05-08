# src/models.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class UrgencyModel(nn.Module):
    def __init__(self, input_size, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()   # outputs in (0,1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_save(patients_df: pd.DataFrame,
                   severity_map=None,
                   epochs: int = 200,
                   lr: float = 1e-2,
                   out_path: str = "urgency_model.pt"):
    """
    Trains a tiny model to map care_profile → urgency and saves it.
    severity_map: dict mapping care_profile → [0.0,1.0] target urgency.
    """
    # default two-tier map if none provided
    if severity_map is None:
        severity_map = {
            "basic_assist":     0.3,
            "mobility_assist":  0.6,
            "fall_risk":        1.0
        }

    # extract care_profile as a categorical
    cats = patients_df["care_profile"].astype("category")
    patients_df["__code"] = cats.cat.codes
    n_cats = len(cats.cat.categories)

    # build training tensors
    X = torch.eye(n_cats)[patients_df["__code"].to_numpy()]    # one-hot
    y = torch.tensor(
        patients_df["care_profile"].map(severity_map).to_numpy(),
        dtype=torch.float,
    )

    # model + optimizer + loss
    model = UrgencyModel(input_size=n_cats)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # train loop
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # save state + category list
    torch.save({
        "state_dict": model.state_dict(),
        "categories": cats.cat.categories.tolist()
    }, out_path)

    print(f"Trained urgency model saved to {out_path}")
    return out_path
