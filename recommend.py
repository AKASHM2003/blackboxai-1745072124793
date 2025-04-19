from data_ingestion import DataIngestion
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedRecommender
import config

class RecommenderSystem:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.spark = self.ingestion.spark
        self.cf = CollaborativeFiltering(self.spark)
        self.cb = ContentBasedRecommender()

        # Load data
        self.purchase_df = self.ingestion.get_spark_purchase_df()
        self.cf.train(self.purchase_df)
        self.cb.fit()

    def recommend(self, user_id, top_n=config.TOP_N_RECOMMENDATIONS):
        # Get collaborative filtering recommendations
        cf_recs = self.cf.recommend_for_user(user_id, top_n)

        # Get user's purchased products
        purchase_pd = self.purchase_df.toPandas()
        user_purchases = purchase_pd[purchase_pd['user_id'] == user_id]['product_id'].unique()

        # Get content-based recommendations for each purchased product
        cb_recs = []
        for product_id in user_purchases:
            cb_recs.extend(self.cb.recommend_similar_products(product_id, top_n))

        # Combine and deduplicate recommendations
        combined_recs = set([rec[0] for rec in cf_recs] + cb_recs)

        # Return top N recommendations
        return list(combined_recs)[:top_n]

if __name__ == "__main__":
    recommender = RecommenderSystem()
    user_id = 1  # example user id
    recommendations = recommender.recommend(user_id)
    print(f"Combined recommendations for user {user_id}: {recommendations}")
