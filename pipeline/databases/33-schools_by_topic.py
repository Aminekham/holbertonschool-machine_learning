#!/usr/bin/env python3
"""
classifying schools based on topic
"""
def schools_by_topic(mongo_collection, topic):
    """
    using the find method
    """
    return mongo_collection.find({"topics": topic})
