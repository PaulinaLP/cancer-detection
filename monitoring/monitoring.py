from datetime import datetime

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    ColumnSummaryMetric,
    ColumnQuantileMetric,
    ColumnDistributionMetric,
    DatasetMissingValuesMetric
)
from evidently.ui.workspace import Workspace
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.ui.dashboards import (
    PlotType,
    CounterAgg,
    PanelValue,
    HistBarMode,
    ReportFilter,
    DashboardPanelPlot,
    DashboardPanelCounter,
    DashboardPanelDistribution
)
from evidently.renderers.html_widgets import WidgetSize


def add_new_report(batch_path, project_id, report_date):
    reference_df = pd.read_csv("input/ref_df.csv")
    comparision_df = pd.read_csv(batch_path)
    numerical_columns = list(reference_df.select_dtypes(include=['number']).columns)
    categorical_columns = ["sex", "anatom_site_general"]

    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=numerical_columns,
        categorical_features=categorical_columns,
    )

    ws = Workspace("workspace")

    regular_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="age_approx", stattest="wasserstein"),
            ColumnDistributionMetric(column_name='anatom_site_general'),
            ColumnQuantileMetric(column_name='tbp_lv_area_perim_ratio', quantile=0.5),
            ColumnQuantileMetric(column_name='age_approx', quantile=0.5),
            ColumnSummaryMetric(column_name="clin_size_long_diam_mm"),
        ],
        timestamp=report_date,
    )

    ws.add_report(project_id, regular_report)
