# E-Commerce Recommender System

## Overview
This project implements an E-Commerce Recommender System that provides personalized product recommendations by analyzing user behavior and purchase history. It uses a hybrid approach combining collaborative filtering and content-based recommendation techniques.

## Technology Stack
- Apache Spark (PySpark) for collaborative filtering using ALS algorithm
- Python with Scikit-learn for content-based recommendations
- MongoDB Atlas for storing user, purchase, and product data
- Pandas and NumPy for data manipulation

## Project Structure
- `config.py`: Configuration file for MongoDB connection and Spark settings
- `data_ingestion.py`: Extracts and preprocesses data from MongoDB Atlas
- `collaborative_filtering.py`: Implements collaborative filtering using Spark MLlib ALS
- `content_based.py`: Implements content-based recommendation using TF-IDF and cosine similarity
- `recommend.py`: Unified interface combining collaborative and content-based recommendations
- `requirements.txt`: Python dependencies

## Setup Instructions
1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Update `config.py` with your MongoDB Atlas connection string and database details.
4. Ensure you have Java installed for Spark to run.
5. Run the recommender system:
   ```
   python recommend.py
   ```

## How It Works
- **Collaborative Filtering:** Uses user-item interaction data to find latent factors and recommend products based on similar user preferences.
- **Content-Based Filtering:** Uses product descriptions to recommend similar products based on content similarity.
- **Hybrid Approach:** Combines both methods to provide more accurate and diverse recommendations.

## Learning Outcomes
- Understanding of collaborative filtering using Spark MLlib ALS.
- Implementation of content-based recommendation using Scikit-learn.
- Integration of MongoDB Atlas with Python for data ingestion.
- Building a hybrid recommendation system.

## Notes
- Ensure MongoDB Atlas collections (`users`, `purchases`, `products`) are populated with relevant data.
- Modify parameters in `config.py` to tune recommendation behavior.
- Logging and error handling can be extended as needed.

## License
This project is open source and free to use.
