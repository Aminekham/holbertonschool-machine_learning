#!/usr/bin/env python3
"""
changing school topics in a document
"""
def update_topics(mongo_collection, name, topics):
    """
    updating based on the name
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
