#!/usr/bin/env python3
"""
inserting a document into a collection
"""
def insert_school(mongo_collection, **kwargs):
    """
    using pymongo to insert
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
