def test_scale_features():
    import numpy as np
    from src.detectors.kmeans import scale_features

    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_val   = np.array([[5.0, 6.0]])
    X_test  = np.array([[7.0, 8.0]])

    scaler, X_train_s, X_val_s, X_test_s = scale_features(
        X_train, X_val, X_test
    )

    assert X_train_s.shape == X_train.shape
    assert X_val_s.shape == X_val.shape
    assert X_test_s.shape == X_test.shape

    # train should be ~zero mean
    np.testing.assert_allclose(X_train_s.mean(axis=0), [0.0, 0.0], atol=1e-7)


def test_fit_kmeans():
    import numpy as np
    from src.detectors.kmeans import fit_kmeans

    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.1]
    ])

    kmeans = fit_kmeans(X, k=2, random_state=42)

    assert kmeans.n_clusters == 2
    assert hasattr(kmeans, "cluster_centers_")



def test_compute_kmeans_distance():
    import numpy as np
    from src.detectors.kmeans import compute_kmeans_distance, fit_kmeans

    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.1]
    ])

    kmeans = fit_kmeans(X, k=2, random_state=42)

    distances, labels = compute_kmeans_distance(X, kmeans)

    assert len(distances) == len(X)
    assert len(labels) == len(X)
    assert np.all(distances >= 0)




def test_compute_cluster_thresholds():
    import numpy as np
    from src.detectors.kmeans import compute_cluster_thresholds

    distances = np.array([0.2, 0.3, 1.5, 1.7])
    labels    = np.array([0,   0,   1,   1])

    thresholds = compute_cluster_thresholds(distances, labels, percentile=90)

    assert set(thresholds.keys()) == {0, 1}
    assert thresholds[0] >= 0.2
    assert thresholds[1] >= 1.5




def test_detect_kmeans_anomaly():
    import numpy as np
    import pandas as pd
    from src.detectors.kmeans import detect_kmeans_anomaly

    distances = np.array([0.2, 2.5, 0.3])
    labels    = np.array([0,   0,   1])

    thresholds = {
        0: 1.0,
        1: 0.8
    }

    meta = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "ticker": ["AAPL", "AAPL", "MSFT"]
    })

    df = detect_kmeans_anomaly(distances, labels, thresholds, meta)

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["anomaly_flag"] == 1
    assert df.iloc[0]["distance"] == 2.5
