import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import config
from pymongo import MongoClient

class ContentBasedRecommender:
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DB_NAME]
        self.products_df = self.load_product_data()
        self.tfidf_matrix = None
        self.tfidf = TfidfVectorizer(stop_words='english')

    def load_product_data(self):
        product_cursor = self.db[config.MONGODB_PRODUCT_COLLECTION].find()
        product_list = list(product_cursor)
        df = pd.DataFrame(product_list)
        if 'description' not in df.columns:
            df['description'] = ''
        return df

    def fit(self):
        self.tfidf_matrix = self.tfidf.fit_transform(self.products_df['description'])

    def recommend_similar_products(self, product_id, top_n=config.TOP_N_RECOMMENDATIONS):
        if self.tfidf_matrix is None:
            self.fit()
        try:
            idx = self.products_df.index[self.products_df['_id'] == product_id][0]
        except IndexError:
            return []
        cosine_similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_n-2:-1]
        recommended_ids = self.products_df.iloc[related_docs_indices]['_id'].tolist()
        recommended_ids = [pid for pid in recommended_ids if pid != product_id]
        return recommended_ids[:top_n]

if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.fit()
    example_product_id = None
    if not recommender.products_df.empty:
        example_product_id = recommender.products_df['_id'].iloc[0]
    if example_product_id:
        recommendations = recommender.recommend_similar_products(example_product_id)
        print(f"Content-based recommendations for product {example_product_id}: {recommendations}")
    else:
        print("No product data available for content-based recommendations.")
