#!/usr/bin/env python3
"""
everything is documented up here
"""
import requests


if __name__ == '__main__':
    """
    and down here also
    """
    launches = requests.get("https://api.spacexdata.com/v4/launches/upcoming").json()
    closest_launch = launches[0]
    for launch in launches:
        if launch["date_unix"] < closest_launch["date_unix"]:
            closest_launch = launch
    rocket = requests.get("https://api.spacexdata.com/v4/rockets/{}".format(closest_launch["rocket"])).json()["name"]
    launch_pad = requests.get("https://api.spacexdata.com/v4/launchpads/{}".format(closest_launch["launchpad"])).json()["name"]
    launch_pad_loc = requests.get("https://api.spacexdata.com/v4/launchpads/{}".format(closest_launch["launchpad"])).json()["locality"]
    print("{} ({}) {} - {} ({})".format(closest_launch["name"], closest_launch["date_local"], rocket, launch_pad, launch_pad_loc))
