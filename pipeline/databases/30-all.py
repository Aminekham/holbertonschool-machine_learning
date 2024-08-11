#!/usr/bin/env python3
"""
the code documentation for looping through a mongodb
collection and listing them
"""
def list_all(mongo_collection):
    """Listing docs in a collection"""
    if mongo_collection is None:
        return []
    documents = mongo_collection.find()
    return list(documents)
