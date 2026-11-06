import numpy as np

from src.backtest import walk_forward_splits
from src.train import persist_models, split_data, train_models


class TestSplitData:
    def test_temporal_order_preserved(self, features_df):
        X_train, X_test, y_train, y_test = split_data(features_df, train_size=0.8)
        assert X_train.index.max() < X_test.index.min(), \
            "train must end before test starts (no shuffling)"
        assert len(X_train) == int(len(features_df) * 0.8)

    def test_label_alignment(self, features_df):
        _, _, y_train, y_test = split_data(features_df, train_size=0.7)
        X_all = features_df["Target"]
        assert (y_train.values == X_all.loc[y_train.index].values).all()
        assert (y_test.values == X_all.loc[y_test.index].values).all()


class TestTrainModels:
    def test_three_models_fitted(self, features_df):
        X_train, X_test, y_train, _ = split_data(features_df, train_size=0.8)
        models = train_models(X_train, y_train)
        assert set(models) == {"Logistic_Regression", "Random_Forest", "XGBoost"}
        for name, model in models.items():
            preds = model.predict(X_test)
            assert len(preds) == len(X_test), f"{name} failed to predict"
            assert set(np.unique(preds)).issubset({0, 1})


class TestPersistModels:
    def test_persists_reloadable_joblib_artifacts(self, features_df, tmp_path):
        import joblib

        X_train, X_test, y_train, _ = split_data(features_df, train_size=0.8)
        models = train_models(X_train, y_train)

        paths = persist_models(models, "aapl", out_dir=tmp_path / "models")

        assert len(paths) == 3
        assert {p.name for p in paths} == {
            "AAPL_Logistic_Regression.joblib",
            "AAPL_Random_Forest.joblib",
            "AAPL_XGBoost.joblib",
        }
        for path in paths:
            assert path.exists() and path.stat().st_size > 0
        # round-trip: a loaded artifact must predict on the same schema
        reloaded = joblib.load(paths[1])
        preds = reloaded.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})
