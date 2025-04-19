from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import config

class CollaborativeFiltering:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.als = ALS(
            userCol="user_id",
            itemCol="product_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=False,
            rank=10,
            maxIter=10,
            regParam=0.1
        )
        self.model = None

    def train(self, ratings_df):
        self.model = self.als.fit(ratings_df)

    def evaluate(self, ratings_df):
        if self.model is None:
            raise Exception("Model not trained yet")
        predictions = self.model.transform(ratings_df)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        return rmse

    def recommend_for_user(self, user_id, num_recommendations=config.TOP_N_RECOMMENDATIONS):
        if self.model is None:
            raise Exception("Model not trained yet")
        user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
        recommendations = self.model.recommendForUserSubset(user_df, num_recommendations)
        recs = recommendations.collect()
        if recs:
            return [(row.product_id, row.rating) for row in recs[0].recommendations]
        else:
            return []

if __name__ == "__main__":
    from data_ingestion import DataIngestion

    ingestion = DataIngestion()
    spark_df = ingestion.get_spark_purchase_df()

    cf = CollaborativeFiltering(ingestion.spark)
    cf.train(spark_df)
    rmse = cf.evaluate(spark_df)
    print(f"Collaborative Filtering Model RMSE: {rmse}")

    user_id = 1  # example user id
    recommendations = cf.recommend_for_user(user_id)
    print(f"Top recommendations for user {user_id}: {recommendations}")
