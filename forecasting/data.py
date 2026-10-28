"""Load the M5 retail sales (Walmart) and aggregate them to the level studied in the benchmark."""
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

# Public mirror of the M5 competition files (Kaggle requires a login).
M5_ZIP_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
RAW_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_m5_zip_if_missing() -> Path:
    zip_path = RAW_DATA_DIRECTORY / "m5.zip"
    if not zip_path.exists():
        RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(M5_ZIP_URL, zip_path)
    return zip_path


def load_store_department_daily_sales() -> pd.DataFrame:
    """Daily unit sales summed per store and department: 10 stores x 7 departments = 70 series
    (level 9 of the M5 hierarchy). Rows are series ids such as "CA_1_FOODS_3", columns are dates."""
    zip_path = download_m5_zip_if_missing()
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("sales_train_evaluation.csv") as item_sales_file:
            item_level_sales = pd.read_csv(item_sales_file)
        with archive.open("calendar.csv") as calendar_file:
            calendar = pd.read_csv(calendar_file, usecols=["date"])

    day_columns = [column for column in item_level_sales.columns if column.startswith("d_")]
    store_department_sales = item_level_sales.groupby(["store_id", "dept_id"])[day_columns].sum()
    store_department_sales.index = [f"{store}_{department}" for store, department in store_department_sales.index]
    # This mirror's calendar has no "d" column: row k (0-based) is day d_{k+1}.
    day_numbers = [int(column.removeprefix("d_")) for column in day_columns]
    store_department_sales.columns = pd.to_datetime(calendar["date"].iloc[[number - 1 for number in day_numbers]].to_numpy())
    return store_department_sales
