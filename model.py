from torch import nn


class Model_Linear(nn.Module):

    def __init__(self) -> None:

        super(Model_Linear, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 60),
        )

        self.decoder = nn.Sequential(
            nn.Linear(60, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 28 * 28), nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
