from torch import nn


class EmotionCNN(nn.Module):

    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.features = nn.Sequential(

            # First Convolution Block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            # 7 emotion classes
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    model = EmotionCNN()

    print("CNN Model Created Successfully!")
    print(model)