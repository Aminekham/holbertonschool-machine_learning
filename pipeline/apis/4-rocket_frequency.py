#!/usr/bin/env python3
"""

"""
import requests


if __name__ == '__main__':
    rockets = {}
    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    for launch in launches:
        if launch["rocket"] not in rockets.keys():
            rocket_name = requests.get("https://api.spacexdata.com/v4/rockets/{}".format(launch["rocket"])).json()["name"]
            rockets[launch["rocket"]] = [rocket_name, 1]
        elif launch["rocket"] in rockets.keys():
            rockets[launch["rocket"]][1] += 1
    for rocket in rockets.keys():
        print("{}: {}\n".format(rockets[rocket][0], rockets[rocket][1]))
