import pandas as pd
import os

df_train = pd.read_csv( "airflow/input/train-metadata.csv")
test_col=["isic_id", "patient_id", "age_approx", "sex", "anatom_site_general", 
        "clin_size_long_diam_mm", "image_type", "tbp_tile_type", "tbp_lv_A", "tbp_lv_Aext", 
        "tbp_lv_B", "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext", "tbp_lv_H", "tbp_lv_Hext", 
        "tbp_lv_L", "tbp_lv_Lext", "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", 
        "tbp_lv_color_std_mean", "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", 
        "tbp_lv_deltaLBnorm", "tbp_lv_eccentricity", "tbp_lv_location", 
        "tbp_lv_location_simple", "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence", 
        "tbp_lv_norm_border", "tbp_lv_norm_color", "tbp_lv_perimeterMM", 
        "tbp_lv_radial_color_std_max", "tbp_lv_stdL", "tbp_lv_stdLExt", 
        "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle", "tbp_lv_x", "tbp_lv_y", "tbp_lv_z", 
        "attribution", "copyright_license"
    ]

df_train=df_train[test_col]   

for i in range (0,40000,10000):
    batch = df_train.iloc[i:(i+10000),:]
    if i==0:
       batch.to_csv("monitoring/input/ref_df.csv", index=False)
    else: 
        batch.to_csv("monitoring/input/batch"+str(int(i/10000))+".csv", index=False)
   