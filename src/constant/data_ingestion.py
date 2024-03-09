import pandas as pd
import urllib.request as request
import os
from pathlib import Path
import zipfile
from src.logger import logging
from src.utils.common import get_size
from src.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config



    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename,header = request.urlretrieve(
                url = self.config.source_URL,
                filename=self.config.local_data_file
            )
            logging.info("f{filename} download! with following info: \n{header}")
        else:
            logging.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
            