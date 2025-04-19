# Configuration file for E-Commerce Recommender System

# MongoDB Atlas connection string
MONGODB_URI = "your_mongodb_atlas_connection_string_here"
MONGODB_DB_NAME = "ecommerce_db"
MONGODB_USER_COLLECTION = "users"
MONGODB_PURCHASE_COLLECTION = "purchases"
MONGODB_PRODUCT_COLLECTION = "products"

# Spark configuration
SPARK_APP_NAME = "ECommerceRecommender"
SPARK_MASTER = "local[*]"

# Recommendation parameters
TOP_N_RECOMMENDATIONS = 10
