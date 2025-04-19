from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import config

class DataIngestion:
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DB_NAME]
        self.spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .config("spark.driver.host", "127.0.0.1") \
            .getOrCreate()

    def load_user_data(self):
        user_cursor = self.db[config.MONGODB_USER_COLLECTION].find()
        user_list = list(user_cursor)
        return pd.DataFrame(user_list)

    def load_purchase_data(self):
        purchase_cursor = self.db[config.MONGODB_PURCHASE_COLLECTION].find()
        purchase_list = list(purchase_cursor)
        return pd.DataFrame(purchase_list)

    def load_product_data(self):
        product_cursor = self.db[config.MONGODB_PRODUCT_COLLECTION].find()
        product_list = list(product_cursor)
        return pd.DataFrame(product_list)

    def get_spark_purchase_df(self):
        purchase_df = self.load_purchase_data()
        if purchase_df.empty:
            return self.spark.createDataFrame([], schema=None)
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = self.spark.createDataFrame(purchase_df)
        # Basic preprocessing: drop rows with null user or product ids
        spark_df = spark_df.dropna(subset=["user_id", "product_id"])
        return spark_df

if __name__ == "__main__":
    ingestion = DataIngestion()
    print("Users Data Sample:")
    print(ingestion.load_user_data().head())
    print("Purchases Data Sample:")
    print(ingestion.load_purchase_data().head())
    print("Products Data Sample:")
    print(ingestion.load_product_data().head())
