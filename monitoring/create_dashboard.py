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

reference_df = pd.read_csv("input/ref_df.csv")
comparision_df = pd.read_csv("input/batch1.csv")

numerical_columns = list(reference_df.select_dtypes(include=['number']).columns)
categorical_columns = ["sex", "anatom_site_general"]

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=numerical_columns,
    categorical_features=categorical_columns,
)

ws = Workspace("workspace")
project = ws.create_project("Cancer Detection Dashboard")
project.description = (
    "This dashboard compares raw data from reference df and incoming data."
)
project.save()

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
    timestamp=datetime.now(),
)

regular_report.run(
    reference_data=reference_df,
    current_data=comparision_df,
    column_mapping=column_mapping,
)

ws.add_report(project.id, regular_report)

# configure the dashboard
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="Cancer Detection data dashboard",
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_missing_values",
                legend="count",
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelCounter(
        title="Share of Drifted Features",
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        value=PanelValue(
            metric_id="DatasetDriftMetric",
            field_path="share_of_drifted_columns",
            legend="share",
        ),
        text="share",
        agg=CounterAgg.LAST,
        size=1,
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Age: Wasserstein drift distance",
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        values=[
            PanelValue(
                metric_id="ColumnDriftMetric",
                metric_args={"column_name.name": "age_approx"},
                field_path=ColumnDriftMetric.fields.drift_score,
                legend="Drift Score",
            ),
        ],
        plot_type=PlotType.BAR,
        size=1,
    )
)

project.save()
