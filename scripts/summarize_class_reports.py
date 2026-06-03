from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def experiment_name(path):
    name = path.name
    if name.endswith("-class-report.xlsx"):
        return name[: -len("-class-report.xlsx")]
    return path.stem


parser = ArgumentParser()
parser.add_argument("--reports", nargs="+", required=True)
parser.add_argument("--out")
args = parser.parse_args()

rows = []
for report_path in args.reports:
    report_path = Path(report_path)
    exp_name = experiment_name(report_path)
    xls = pd.ExcelFile(report_path)
    for dataset in xls.sheet_names:
        df = pd.read_excel(report_path, sheet_name=dataset, index_col=0)
        for row_name, row in df.iterrows():
            for metric, value in row.items():
                rows.append(
                    {
                        "experiment": exp_name,
                        "dataset": dataset,
                        "row": row_name,
                        "metric": metric,
                        "value": value,
                    }
                )

summary = pd.DataFrame(rows)
if args.out:
    summary.to_csv(args.out, index=False)
else:
    print(summary.to_string(index=False))
