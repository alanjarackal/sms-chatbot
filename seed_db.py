import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def seed_database():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri or "your_mongodb_connection_string" in mongo_uri:
        print("Error: MONGO_URI not set or is still a placeholder in .env")
        return

    try:
        client = MongoClient(mongo_uri)
        db = client["nyaya_db"]
        clients_collection = db["clients"]

        # Multiple test clients
        test_clients = [
            {
                "client_id": "rajesh-123",
                "name": "Rajesh Kumar",
                "case_details": "Property dispute regarding ancestral land in Haryana. Ongoing since 2022.",
                "history": "Previous cases related to land encroachment; cleared in 2018.",
                "status": "Active"
            },
            {
                "client_id": "priya-456",
                "name": "Priya Sharma",
                "case_details": "Consumer court case against an e-commerce giant for faulty electronics.",
                "history": "No previous legal history.",
                "status": "Active"
            },
            {
                "client_id": "default_client",
                "name": "Unknown Client",
                "case_details": "General inquiry about Indian Penal Code.",
                "history": "None",
                "status": "Visitor"
            }
        ]

        # Clear and insert
        clients_collection.delete_many({"client_id": {"$in": ["rajesh-123", "priya-456", "default_client"]}})
        result = clients_collection.insert_many(test_clients)
        
        print(f"Database seeded successfully. Inserted {len(result.inserted_ids)} records.")
    
    except Exception as e:
        print(f"Error seeding database: {e}")

if __name__ == "__main__":
    seed_database()
