import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric, ColumnSummaryMetric
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
from evidently.renderers.html_widgets import WidgetSize
from datetime import datetime

reference_df=pd.read_csv("input/ref_df.csv")
comparision_df=pd.read_csv("input/batch1.csv")

numerical_columns = list(reference_df.select_dtypes(include=['number']).columns) 
categorical_columns = ["sex", "anatom_site_general"]  

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=numerical_columns,
    categorical_features=categorical_columns
)

ws = Workspace("workspace")
project = ws.create_project("Cancer Detection 3")
project.description = "This dashboard compares raw data from reference df and incoming data."
project.save()

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
    
ws.add_report(project.id, regular_report)

#configure the dashboard
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="Cancer Detection data dashboard"
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Inference Count",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_rows",
                legend="count"
            ),
        ],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_missing_values",
                legend="count"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Age Quantile",
        values=[
            PanelValue(
                metric_id="ColumnQuantileMetricResultc:age_approx:num:0.5",                
                field_path="current.value",
                legend="Age 0.5 Quantile"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.save()