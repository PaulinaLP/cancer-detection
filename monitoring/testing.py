import datetime

from monitoring import add_new_report

batch_path = "input/batch2.csv"
project_id = "74d9852e-f87f-4535-b6b4-4470b6eda8cb"
report_date = timestamp = datetime.now()

add_new_report(batch_path, project_id, report_date)
