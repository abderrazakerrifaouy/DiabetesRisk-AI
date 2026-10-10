import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer



class DataCleaner:

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.csv_path}")
        self.df = pd.read_csv(self.csv_path )
        return self.df
    
    def _check_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Le DataFrame n'est pas chargé. Veuillez appeler la méthode 'load()' d'abord.")
        return self.df

    def drop_index_column(self) -> pd.DataFrame:

        df = self._check_loaded()
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    
    def mark_impossible_zeros(self) -> pd.DataFrame:
        
        df = self._check_loaded()
        impossible_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in impossible_zero_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        return df

    def detect_outliers_iqr(self) -> pd.DataFrame:
        df = self._check_loaded()
        numeric_cols = df.select_dtypes(include="number").columns
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        return outlier_mask

    def cap_or_flag_outliers(self):
        df = self._check_loaded()
        outlier_mask = self.detect_outliers_iqr()
        capped_df = df.copy()

        for col in outlier_mask.columns:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                capped_df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return capped_df, outlier_mask

    def impute_missing(self):
        df = self._check_loaded()
        imputer = KNNImputer(n_neighbors=3)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df_imputed

    def clean_data(self) -> pd.DataFrame:
        self.df = self._check_loaded()
        self.df = self.drop_index_column()
        self.df = self.mark_impossible_zeros()
        self.df, outlier_mask = self.cap_or_flag_outliers()
        df_imputed = self.impute_missing()
        return df_imputed
    
    def save_cleaned_data(self, output_path: str | Path) -> None:
        df_cleaned = self.clean_data()
        df_cleaned.to_csv(output_path, index=False)

    def run_cleaning_pipeline(self, output_path: str | Path) -> None:
        self.load()
        self.clean_data()
        self.save_cleaned_data(output_path)

if __name__ == "__main__":
    cleaner = DataCleaner("C:/Users/safiy/Desktop/les brief/DiabetesRisk-AI/data/raw/dataset-diabete.csv")
    cleaner.run_cleaning_pipeline("data/processed/cleaned_data_diabetes.csv")
