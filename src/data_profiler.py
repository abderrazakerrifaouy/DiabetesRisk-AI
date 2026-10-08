from pathlib import Path

import pandas as pd


class DataProfiler:

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def shape_info(self) -> dict:

        df = self._check_loaded()
        return {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

    def missing_values(self) -> pd.Series:

        df = self._check_loaded()
        return df.isna().sum()

    def duplicates(self) -> int:

        df = self._check_loaded()
        return int(df.duplicated().sum())

    def zero_counts(self, columns: list[str] | None = None) -> pd.Series:

        df = self._check_loaded()
        numeric = df[columns] if columns else df.select_dtypes(include="number")
        return (numeric == 0).sum()

    def numeric_distribution(self) -> pd.DataFrame:

        df = self._check_loaded()
        return df.select_dtypes(include="number").describe().T

    def outlier_counts_iqr(self, columns: list[str] | None = None) -> pd.Series:

        df = self._check_loaded()
        cols = columns or list(df.select_dtypes(include="number").columns)
        counts = {}
        for col in cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
        return pd.Series(counts, name="outliers_iqr")

    def correlations(self) -> pd.DataFrame:

        df = self._check_loaded()
        return df.select_dtypes(include="number").corr()

    def report(self) -> None:
        
        info = self.shape_info()
        print(f"Dimensions : {info['n_rows']} lignes x {info['n_cols']} colonnes")
        print(f"Colonnes   : {info['columns']}")
        print("\nTypes :")
        print(pd.Series(info["dtypes"]))
        print("\nValeurs manquantes (NaN) :")
        print(self.missing_values())
        print(f"\nDoublons : {self.duplicates()}")
        print("\nZéros par colonne :")
        print(self.zero_counts())
        print("\nOutliers (IQR) :")
        print(self.outlier_counts_iqr())
        print("\nDistribution :")
        print(self.numeric_distribution())

    # ------------------------------------------------------------------ interne
    def _check_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Appelez load() avant d'utiliser cette méthode.")
        return self.df
