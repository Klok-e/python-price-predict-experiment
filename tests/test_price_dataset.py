import numpy as np
import pandas as pd
import pytest

from train_supervised import PriceDataset, calculate_sample_weights


@pytest.fixture
def synthetic_tickers():
    """
    Builds two tiny “ticker” tuples of the form
        (scaled_features_df, <placeholder>, labels_df)
    that the training utilities expect.
    """
    # --- ticker 1 ------------------------------------------------------------
    data_1 = pd.DataFrame(
        np.arange(20, dtype=np.float32).reshape(10, 2), columns=["f1", "f2"]
    )
    # alternating 0 / 1 so that both classes appear
    labels_1 = pd.DataFrame({"label": [0, 1] * 5}, dtype=np.float32)

    # --- ticker 2 ------------------------------------------------------------
    data_2 = pd.DataFrame(
        np.arange(16, dtype=np.float32).reshape(8, 2), columns=["f1", "f2"]
    )
    # inverse pattern to ticker-1
    labels_2 = pd.DataFrame({"label": [1, 0] * 4}, dtype=np.float32)

    # Each tuple mimics the `(scaled_df, something, labels_df)` structure
    return [
        (data_1, None, labels_1),
        (data_2, None, labels_2),
    ]


# --------------------------------------------------------------------------- #
#  PriceDataset                                                               #
# --------------------------------------------------------------------------- #
def test_dataset_length_and_stride(synthetic_tickers):
    window_size, stride = 3, 2
    ds = PriceDataset(synthetic_tickers, window_size=window_size, stride=stride)

    # ticker-1: (10-3)//2 + 1  = 4 samples
    # ticker-2: (8-3)//2  + 1  = 3 samples
    assert len(ds) == 7, "Total sample count across all tickers is incorrect"


def test_dataset_windows_correct_positions(synthetic_tickers):
    window_size, stride = 3, 2
    ds = PriceDataset(synthetic_tickers, window_size=window_size, stride=stride)

    # Helper: first row of window (index 0) should correspond to the starting
    # position of that window in the original dataframe
    expected_starts = [0, 2, 4, 6, 0, 2, 4]  # indices per ticker/stride
    for i, start_row in enumerate(expected_starts):
        window, label = ds[i]

        # shape checks
        assert window.shape == (window_size, 2)
        assert label.shape == (1,)

        # the very first feature value in the window should equal the original
        # dataframe’s first value at `start_row`
        if i < 4:
            original_df = synthetic_tickers[0][0]
            original_labels = synthetic_tickers[0][2]
        else:
            original_df = synthetic_tickers[1][0]
            original_labels = synthetic_tickers[1][2]

        np.testing.assert_array_equal(window[0], original_df.iloc[start_row].to_numpy())
        assert label[0] == original_labels.iloc[start_row + window_size - 1, 0]


# --------------------------------------------------------------------------- #
#  calculate_sample_weights                                                   #
# --------------------------------------------------------------------------- #
def test_sample_weights_mirror_class_imbalance(synthetic_tickers):
    window_size, stride = 3, 2
    weights = calculate_sample_weights(
        synthetic_tickers, window_size=window_size, stride=stride
    )

    # One weight per produced sample
    assert len(weights) == 7

    # Class distribution: 4 x class-0, 3 x class-1  ⇒ minority class weight larger
    weight_class0 = weights[0]          # any element with label 0
    weight_class1 = weights[-1]         # any element with label 1
    assert weight_class1 > weight_class0, "Minority class should receive higher weight"

    # Sanity: weights follow inverse-frequency rule  (N / count[class])
    unique, counts = np.unique(
        np.concatenate(
            [t[2].iloc[window_size - 1 :: stride].to_numpy().flatten()  # labels per ticker
             for t in synthetic_tickers]
        ),
        return_counts=True,
    )
    expected = {cls: len(weights) / cnt for cls, cnt in zip(unique, counts)}
    for w, lbl in zip(weights, np.repeat(unique, counts)):
        assert np.isclose(w, expected[lbl])
