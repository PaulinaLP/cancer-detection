import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric, ColumnSummaryMetric
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace
from datetime import datetime

reference_df=pd.read_csv("input/ref_df.csv")
comparision_df=pd.read_csv("input/batch3.csv")
numerical_columns = list(reference_df.select_dtypes(include=['number']).columns) 
categorical_columns = ["sex", "anatom_site_general"]  

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=numerical_columns,
    categorical_features=categorical_columns
)

ws = Workspace("workspace")

regular_report = Report(
        metrics=[       
            DataQualityPreset(), 
            ColumnQuantileMetric(column_name='tbp_lv_area_perim_ratio', quantile=0.2),
            ColumnQuantileMetric(column_name='age_approx', quantile=0.5)
        ],
        timestamp=datetime.now()
    )

regular_report.run(reference_data=reference_df,
                      current_data=comparision_df,
                      column_mapping=column_mapping)

ws.add_report("a6c4192c-f3c1-4f52-b7a5-74a4fd963242", regular_report)
