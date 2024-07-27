#!/usr/bin/env python3
"""
the documentation check
"""
import requests
import sys
import time

if __name__ == '__main__':
    """
    main function in python
    """
    api_url = sys.argv[1]
    response = requests.get(api_url)
    if response.status_code == 200:
        user_data = response.json()
        print(user_data["location"])
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = response.headers.get("X-Ratelimit-Reset")
        reset_time = int(reset_time)
        current_time = int(time.time())
        wait_time = (reset_time - current_time) // 60
        print("Reset in {} min".format(wait_time))
