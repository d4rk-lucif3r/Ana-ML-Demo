import os
import pandas as pd
import pickle
import gridfs
from pymongo import MongoClient
from io import BytesIO

class MongoDBClient:
    def __init__(self):
        # Use environment variable for connection string
        self.client = MongoClient(os.getenv('MONGODB_URI'))
        self.db = self.client['healthcare_camp']
        self.fs = gridfs.GridFS(self.db)
    
    def store_dataframe(self, df, collection_name):
        """Store DataFrame in MongoDB"""
        self.db[collection_name].delete_many({})  # Clear existing data
        self.db[collection_name].insert_many(df.to_dict('records'))
    
    def load_dataframe(self, collection_name):
        """Load DataFrame from MongoDB"""
        data = list(self.db[collection_name].find({}, {'_id': 0}))
        return pd.DataFrame(data)
    
    def store_model(self, model, scaler, features, model_name='latest_model'):
        """Store model, scaler, and features in GridFS"""
        # Serialize the objects
        model_bytes = BytesIO()
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': features
        }, model_bytes)
        model_bytes.seek(0)
        
        # Remove existing model if it exists
        existing = self.db.fs.files.find_one({'filename': model_name})
        if existing:
            self.fs.delete(existing['_id'])
        
        # Store new model
        self.fs.put(model_bytes.getvalue(), filename=model_name)
    
    def load_model(self, model_name='latest_model'):
        """Load model, scaler, and features from GridFS"""
        try:
            # Find and read the file
            grid_out = self.fs.find_one({'filename': model_name})
            if grid_out:
                model_data = pickle.loads(grid_out.read())
                return model_data['model'], model_data['scaler'], model_data['features']
        except Exception as e:
            print(f"Error loading model: {e}")
        return None, None, None

# Initialize global client
mongodb_client = MongoDBClient()
