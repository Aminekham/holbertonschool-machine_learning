#!/usr/bin/env python3
"""
test doc
"""
import requests


def sentientPlanets():
    """
    test doc 2.0
    """
    url = "https://swapi-api.hbtn.io/api/species"
    sentient_planets = []
    while url:
        response = requests.get(url)
        species_data = response.json()
        for specie in species_data["results"]:
            if specie["designation"] == "sentient" or specie["classification"]:
                planet = specie["homeworld"]
                if planet is None:
                    continue
                planet = requests.get(planet).json()
                sentient_planets.append(planet["name"])
        url = species_data["next"]
        if url is None:
            break
    return sentient_planets
