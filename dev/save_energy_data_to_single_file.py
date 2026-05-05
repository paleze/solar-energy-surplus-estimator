
import logging
import pandas as pd
from pathlib import Path


def fecth_data(directory: Path, pattern="*.csv", concat=True, add_filename_col=True, sort=True) -> pd.DataFrame:
    """
    Download files from an S3 bucket and Load CSV files from `directory`.

    Args:
    - endpoint (str): url to folder containing CSV files
    - pattern (str): glob pattern to match files (default "*.csv").
    - concat (bool): if True, return a single concatenated DataFrame; if False, return a list of DataFrames.
    - add_filename_col (bool): if True, add a column '_source_file' with the filename for each row.
    - sort (bool): if True, sort files by name before loading.

    Returns:
    - pandas.DataFrame or list[pandas.DataFrame]
    """

    
    # Load files from the local directory into a DataFrame
    
    files = list(directory.glob(pattern))
    if sort:
        files = sorted(files)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep='\t', encoding='utf-8')
        except Exception as e:
            # re-raise with context
            raise RuntimeError(f"Failed to read {f}: {e}") from e
        if add_filename_col:
            df["_source_file"] = f.name
        dfs.append(df)

    if dfs.count==0:
        raise RuntimeError(f"Error creating data frame")

    if concat:
        dfs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    logging.info(f"Fetched data frame has {len(dfs)} rows and {len(dfs.columns)} columns")

    return dfs


if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    df = fecth_data(Path("../minio/source-data"), pattern="*.csv", concat=True, add_filename_col=False, sort=True)
    
    df_save = df[['timestamp','solar_energy_exported','consumer_energy_imported_from_grid','consumer_energy_imported_from_solar','consumer_energy_imported_from_battery']]
    df_save.to_csv("energy-data.csv", encoding='utf-8', index=False, header=True)