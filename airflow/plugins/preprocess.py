import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold

# creating a variable with columns that appear both in test and train
test_and_target = [
    "target",
    "isic_id",
    "patient_id",
    "age_approx",
    "sex",
    "anatom_site_general",
    "clin_size_long_diam_mm",
    "image_type",
    "tbp_tile_type",
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_location",
    "tbp_lv_location_simple",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
    "attribution",
    "copyright_license",
]


def prepare_train(df):
    df_train_filtered = df[test_and_target]
    df_train_filtered = df_train_filtered[
        df_train_filtered['anatom_site_general'].notna()
    ]
    df_train_filtered.reset_index(drop=True, inplace=True)
    return df_train_filtered


def feature_engineering(df):
    # Taken from https://www.kaggle.com/code/snnclsr/tabular-ensemble-lgbm-catboost
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2
    )
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = (
        df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    )
    df["3d_position_distance"] = np.sqrt(
        df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2
    )
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["symmetry_border_consistency"] = (
        df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    )
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (
        df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]
    ) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = (
        df["tbp_lv_deltaA"]
        + df["tbp_lv_deltaB"]
        + df["tbp_lv_deltaL"]
        + df["tbp_lv_deltaLBnorm"]
    )
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt(
        (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
        / 3
    )
    df["color_shape_composite_index"] = (
        df["tbp_lv_color_std_mean"]
        + df["tbp_lv_area_perim_ratio"]
        + df["tbp_lv_symm_2axis"]
    ) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (
        df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]
    ) / 3
    df["symmetry_perimeter_interaction"] = (
        df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    )
    df["comprehensive_lesion_index"] = (
        df["tbp_lv_area_perim_ratio"]
        + df["tbp_lv_eccentricity"]
        + df["tbp_lv_norm_color"]
        + df["tbp_lv_symm_2axis"]
    ) / 4
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"] = (
        df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    )
    df["age_normalized_nevi_confidence"] = (
        df["tbp_lv_nevi_confidence"] / df["age_approx"]
    )
    df["color_asymmetry_index"] = (
        df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]
    )
    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(
        df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2
    )
    df["color_range"] = (
        (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
        + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs()
        + (df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    )
    df["shape_color_consistency"] = (
        df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    )
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (
        2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi)
    )
    df["age_size_symmetry_index"] = (
        df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    )
    return df


def feature_engineering_process(df, num_columns, medians):
    # Handle numerical columns missing values
    for nc in num_columns:
        df[nc] = df[nc].fillna(medians.get(nc, 0))
    # Replacement for 0 to na to avoid incorrect calulation in feature engineering as 0 really mean missing value
    replace_cols = [
        'tbp_lv_color_std_mean',
        'tbp_lv_norm_color',
        'tbp_lv_radial_color_std_max',
    ]
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    # Feature engineering
    df_num = df[num_columns]
    df_num = feature_engineering(df_num)
    numerical_columns_list = list(df_num.columns)
    return df_num, numerical_columns_list


class Preprocessor:
    def __init__(self):
        self.medians = {}
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.encoded_cols = []
        self.numerical_columns = []
        self.cat_cols = ["sex", "anatom_site_general"]

    def fit(self, df):
        # Check if input is a numpy array and convert to DataFrame if necessary
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)

        # Determine numerical columns
        self.numerical_columns = list(df.select_dtypes(include=['number']).columns)
        if "target" in self.numerical_columns:
            self.numerical_columns.remove("target")

        # Compute medians for numerical columns
        num_medians = df[self.numerical_columns].median()
        self.medians = dict(num_medians)

        # Compute median for angle
        if 'tbp_lv_symm_2axis_angle' in df.columns:
            self.medians['tbp_lv_symm_2axis_angle'] = df[
                'tbp_lv_symm_2axis_angle'
            ].median()

        # Fit OneHotEncoder
        df['sex'] = df['sex'].fillna('male')
        df[self.cat_cols] = df[self.cat_cols].fillna('other')
        self.one_hot_encoder.fit(df[self.cat_cols])
        self.encoded_cols = self.one_hot_encoder.get_feature_names_out(self.cat_cols)

        # I have to do feature engineering process in fit as I nedd the result to fit the scaler
        # Feature engineering
        df_copy = df.copy()
        df_copy, numerical_columns_list = feature_engineering_process(
            df_copy, self.numerical_columns, self.medians
        )

        # Scale numerical columns
        self.scaler.fit(df_copy[numerical_columns_list])

        return self

    def transform(self, df):
        # Handle categorical columns missing values
        df['sex'] = df['sex'].fillna('male')
        df[self.cat_cols] = df[self.cat_cols].fillna('other')

        # Specific replacement for 'tbp_lv_symm_2axis_angle' as 0 mean really missing value
        if 'tbp_lv_symm_2axis_angle' in df.columns:
            df['tbp_lv_symm_2axis_angle'] = df['tbp_lv_symm_2axis_angle'].replace(
                0, self.medians['tbp_lv_symm_2axis_angle']
            )

        # Apply one-hot encoding
        X_cat = self.one_hot_encoder.transform(df[self.cat_cols])
        dense_matrix = X_cat.todense()
        df_encoded = pd.DataFrame(dense_matrix, columns=self.encoded_cols)

        # Feature engineering
        df_copy = df.copy()
        df_num, numerical_columns_list = feature_engineering_process(
            df_copy, self.numerical_columns, self.medians
        )

        # Scale numerical columns
        df_num[numerical_columns_list] = self.scaler.transform(
            df_num[numerical_columns_list]
        )

        # Combine encoded and scaled data
        df_final = pd.concat([df_num, df_encoded], axis=1)

        # Add id columns and target back if they exist
        if "isic_id" in df.columns and "patient_id" in df.columns:
            ids = df[["isic_id", "patient_id"]]
            df_final = pd.concat([df_final, ids], axis=1)
        if "target" in df.columns:
            targets = df[["target"]]
            df_final = pd.concat([df_final, targets], axis=1)

        return df_final
