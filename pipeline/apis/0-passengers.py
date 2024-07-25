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
    url = "https://swapi-api.hbtn.io/api/starships"
    while requests.get(url):
        ships = requests.get(url)
        ships = ships.json()
        for ship in ships["results"]:
            passengers = ship["passengers"]
            if passengers == 'n/a' or passengers == 'unknown':
                continue
            if ',' in passengers:
                passengers = passengers.split(',')
                passengers = passengers[0] + passengers[1]
            if float(passengers) >= float(passengerCount):
                ships_list.append(ship["name"])
        if ships["next"] is not None:
            url = ships["next"]
        else:
            break
    return ships_list
