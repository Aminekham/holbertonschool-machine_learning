#!/usr/bin/env python3
"""

"""
import requests

def availableShips(passengerCount):
    """

    """
    ships_list = []
    ships = requests.get("https://swapi-api.hbtn.io/api/starships")
    ships = ships.json()
    for ship in ships["results"]:
        if ship["passengers"] >= passengerCount:
            ships_list.append(ship["name"])
    return ships_list
