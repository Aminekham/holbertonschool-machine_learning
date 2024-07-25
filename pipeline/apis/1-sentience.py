#!/usr/bin/env python3
"""

"""
import requests


def sentientPlanets():
    """
    
    """
    url = "https://swapi-api.hbtn.io/api/planets/"
    planets_names = []
    while 1:
        planets = requests.get(url)
        planets = planets.json()
        for planet in planets["results"]:
            for resident in planet["residents"]:
                resident_info = requests.get(resident)
                resident_info = resident_info.json()
                if len(resident_info["species"]) != 0:
                    specie = requests.get(resident_info["species"][0])
                    specie = specie.json()
                    if (specie["designation"] == "sentient" or specie["classification"] == "sentient") and planet["name"] not in planets_names:
                        planets_names.append(planet["name"])
                else:
                    continue
        if planets["next"] is not None:
            url = planets["next"]
        else:
            break
    return planets_names
