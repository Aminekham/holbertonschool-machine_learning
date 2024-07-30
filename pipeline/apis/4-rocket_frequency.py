#!/usr/bin/env python3
"""
hello this is me doc
"""
import requests


if __name__ == '__main__':
    """
    its me again doc 2.0
    """
    rockets = {}
    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    for launch in launches:
        if launch["rocket"] not in rockets.keys():
            lk = "https://api.spacexdata.com/v4/rockets/"
            rocket_name = requests.get("{}{}".format(lk ,launch["rocket"])).json()["name"]
            rockets[launch["rocket"]] = [rocket_name, 1]
        elif launch["rocket"] in rockets.keys():
            rockets[launch["rocket"]][1] += 1
    rocket_list = []
    for rocket in rockets.keys():
        rocket_list.append(rockets[rocket])
    rocket_list = sorted(rocket_list, key=lambda x: x[1], reverse=True)
    for rocket_pro in rocket_list:
        print("{}: {}".format(rocket_pro[0], rocket_pro[1]))
