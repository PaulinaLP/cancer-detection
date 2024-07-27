import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric, ColumnSummaryMetric
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelDistribution, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
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
project = ws.create_project("Cancer Detection Dashboard 2")
project.description = "This dashboard compares raw data from reference df and incoming data."
project.save()

regular_report = Report(
        metrics=[       
            DataQualityPreset(), 
            DataDriftPreset(),
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
        title="Cancer Detection data dashboard 2"
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
                legend="count"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

p.dashboard.add_panel(
        DashboardPanelDistribution(
            title="Column Distribution: current",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                field_path="ColumnDistributionMetric.fields.current",
                metric_id="ColumnDistributionMetric",
                metric_args={"column_name.name": "age_approx"},
            ),
            barmode = HistBarMode.STACK
        )
    )

project.save()
