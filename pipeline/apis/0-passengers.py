#!/usr/bin/env python3
"""
This is the needed
function to get specific
"""
import requests

def availableShips(passengerCount):
    """
    informations from an API
    using the request library
    """
    ships_list = []
    ships = requests.get("https://swapi-api.hbtn.io/api/starships")
    ships = ships.json()
    for ship in ships["results"]:
        if ship["passengers"] >= passengerCount:
            ships_list.append(ship["name"])
    return ships_list
