from sklearn.datasets import load_digits

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_seeds_data():
    x, y = load_digits(return_X_y=True)

    #scaler = MaxAbsScaler()
    #x_scaled = scaler.fit_transform(x)
    #print(x_scaled.min(), x_scaled.max())
    x_scaled = x * 255/ 16  
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=7777
    )

    train_dataset = [
        (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y, dtype=torch.long).unsqueeze(0),
        )
        for x, y in zip(x_train, y_train)
    ]

    test_dataset = [
        (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y, dtype=torch.long).unsqueeze(0),
        )
        for x, y in zip(x_test, y_test)
    ]

    return train_dataset, test_dataset


def seeds_dataloaders():
    train, test = load_seeds_data()
    loader = {
        "train": train,
        "test": test,
    }
    return loader

ArrayLike = Tuple[Any, Any]



def load_train_test_data_mnist64(
    as_numpy: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Return the Iris dataset as full train/test tensors.

    Parameters
    ----------
    as_numpy:
        If True, convert tensors to numpy arrays before returning.
    """
    dataloaders = seeds_dataloaders()
    # Load the entire split at once.
    train_loader = torch.utils.data.DataLoader(
        dataloaders["train"],
        batch_size=len(dataloaders["train"]),
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataloaders["test"],
        batch_size=len(dataloaders["test"]),
        shuffle=False,
    )

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    # Remove singleton channel dimensions introduced earlier.
    x_train = x_train.squeeze(1)
    x_test = x_test.squeeze(1)
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)

    if as_numpy:
        return (
            (x_train.numpy(), x_test.numpy()),
            (y_train.numpy(), y_test.numpy()),
        )

    return (x_train, x_test), (y_train, y_test)

if __name__ == "__main__":
    (x_train, x_test), (y_train, y_test) = load_train_test_data_mnist64()
    print(x_test[0])
    print(y_test[0])