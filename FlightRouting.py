import pandas as pd
import numpy as np
import random
import math

# Load the datasets
aircraft_data = pd.read_csv("aircraftDataset.csv")
airport_data = pd.read_csv("airportDataset.csv")
weather_data = pd.read_csv("weatherDataset.csv")

# Create separate arrays for each column in each CSV
# For aircraftDataset.csv
aircraft_type_array = aircraft_data['Aircraft Type'].to_numpy()
max_range_array = aircraft_data['Max Range'].to_numpy()
cruise_speed_array = aircraft_data['Cruise Speed'].to_numpy()
fuel_consumption_array = aircraft_data['Fuel Consumption at Cruise'].to_numpy()
icao_codes_aircraft_array = aircraft_data['ICAO CODES'].to_numpy()

# For airportDataset.csv
icao_codes_airport_array = airport_data['ICAO Code'].to_numpy()
latitude_array = airport_data['Latitude'].to_numpy()
longitude_array = airport_data['Longitude'].to_numpy() 
city_airport_array = airport_data['City'].to_numpy()

# For weatherDataset.csv
date_array = weather_data['Date'].to_numpy()
wind_speed_array = weather_data['Wind Speed'].to_numpy()
wind_direction_array = weather_data['Wind Direction'].to_numpy()
city_weather_array = weather_data['City'].to_numpy()

'''
# User input for the departure airport's ICAO code
departure_icao_input = input("Enter the ICAO code of the departure airport: ").upper()
departure_icao = departure_icao_input[:4]

# User input for the destination city
destination_city = input("Enter the name of the destination city: ").title()

# User input for the date of travel
date_of_travel = input("Enter the date of travel (YYYY-MM-DD): ")

'''
def generate_random_route(departure_icao, destination_city, icao_codes_airport_array, city_airport_array):
    # Convert destination city to ICAO code format (first 4 letters, uppercase)
    destination_icao = destination_city[:4].upper()

    # Filter out the departure and destination ICAO codes from the possible stopovers
    possible_stopovers = [icao for icao in icao_codes_airport_array if icao not in [departure_icao, destination_icao]]

    # Generate a random number of stopovers (e.g., between 1 and 3)
    num_stopovers = random.randint(1, 5)

    # Randomly select stopovers
    route_stopovers = random.sample(possible_stopovers, num_stopovers)

    # Construct the final route
    route = [departure_icao] + route_stopovers + [destination_icao]

    return route

def haversine(coord1, coord2):
    # Coordinates in decimal degrees (e.g. 43.60, -79.49)
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Radius of the Earth in km
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Change in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance/10

def calculate_total_distance(route, latitude_array, longitude_array, icao_codes_airport_array):
    total_distance = 0.0
    for i in range(len(route) - 1):
        # Convert city name to ICAO code
        route_departure_icao = route[i][:4].upper()
        arrival_icao = route[i + 1][:4].upper()

        # Find index of departure and arrival airports in the ICAO codes array
        departure_index = list(icao_codes_airport_array).index(route_departure_icao)
        arrival_index = list(icao_codes_airport_array).index(arrival_icao)

        # Get coordinates of departure and arrival airports
        departure_coords = (latitude_array[departure_index], longitude_array[departure_index])
        arrival_coords = (latitude_array[arrival_index], longitude_array[arrival_index])

        # Calculate distance and add to total
        total_distance += haversine(departure_coords, arrival_coords)

    return total_distance

def pick_suitable_aircraft(route, aircraft_type_array, icao_codes_aircraft_array, max_range_array, latitude_array, longitude_array, icao_codes_airport_array):
    # Iterate over each segment of the route
    for i in range(len(route) - 1):
        departure_icao = route[i][:4].upper()
        arrival_icao = route[i + 1][:4].upper()

        # Get the indices of the departure and arrival airports
        departure_index = list(icao_codes_airport_array).index(departure_icao)
        arrival_index = list(icao_codes_airport_array).index(arrival_icao)

        # Calculate the distance between the two airports
        departure_coords = (latitude_array[departure_index], longitude_array[departure_index])
        arrival_coords = (latitude_array[arrival_index], longitude_array[arrival_index])
        segment_distance = haversine(departure_coords, arrival_coords)

        # Filter aircraft that are parked at the departure airport and have sufficient range for this segment
        suitable_aircraft = [aircraft_type_array[j] for j in range(len(aircraft_type_array)) 
                             if icao_codes_aircraft_array[j] == departure_icao 
                             and max_range_array[j] >= segment_distance]

        # If no suitable aircraft is found for any segment, return a failure
        if not suitable_aircraft:
            return "No suitable aircraft found for segment from {} to {}".format(departure_icao, arrival_icao)

        # Randomly select an aircraft from the suitable ones for this segment
        # (You can modify this logic as per your requirement)
        selected_aircraft = random.choice(suitable_aircraft)

        # You might also want to ensure that the same aircraft is used for the entire route
        # (This part of the logic depends on your specific requirements)

    # Return the selected aircraft (assuming the same aircraft is used for the whole route)
    return selected_aircraft



def create_initial_population(num_chromosomes, departure_icao, destination_city, aircraft_type_array, icao_codes_aircraft_array, max_range_array, latitude_array, longitude_array, icao_codes_airport_array, city_airport_array):
    population = []
    for _ in range(num_chromosomes):
        # Generate random route
        route = generate_random_route(departure_icao, destination_city, icao_codes_airport_array, city_airport_array)
        
        # Calculate total distance of the route
        total_distance = calculate_total_distance(route, latitude_array, longitude_array, icao_codes_airport_array)
        
        # Select a suitable aircraft for the route
        aircraft = pick_suitable_aircraft(route, aircraft_type_array, icao_codes_aircraft_array, max_range_array, latitude_array, longitude_array, icao_codes_airport_array)
        
        # Skip chromosome creation if no suitable aircraft is found
        if aircraft == "No suitable aircraft found":
            continue

        # Create chromosome
        chromosome = {
            'route': route,
            'total_distance': total_distance,
            'aircraft': aircraft
        }
        
        population.append(chromosome)
    return population


def fuel_consumption_speed_fitness(route, aircraft, cruise_speed_array, fuel_consumption_array, latitude_array, longitude_array, icao_codes_airport_array):
    aircraft_index = list(aircraft_type_array).index(aircraft)
    cruise_speed = cruise_speed_array[aircraft_index]
    fuel_consumption = fuel_consumption_array[aircraft_index]

    total_fuel = 0
    total_time = 0
    for i in range(len(route) - 1):
        departure_icao = route[i][:4].upper()
        arrival_icao = route[i + 1][:4].upper()

        departure_index = list(icao_codes_airport_array).index(departure_icao)
        arrival_index = list(icao_codes_airport_array).index(arrival_icao)

        departure_coords = (latitude_array[departure_index], longitude_array[departure_index])
        arrival_coords = (latitude_array[arrival_index], longitude_array[arrival_index])

        segment_distance = haversine(departure_coords, arrival_coords)  # Calculate the distance
        segment_time = segment_distance / cruise_speed
        segment_fuel = segment_time * fuel_consumption

        total_fuel += segment_fuel
        total_time += segment_time

    return 1 / (total_fuel + total_time) if total_fuel + total_time > 0 else 0



def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing between two points on the earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    return math.degrees(initial_bearing)

def weather_impact_fitness(route, date_of_travel, weather_data, wind_speed_array, wind_direction_array, city_weather_array, date_array, latitude_array, longitude_array, icao_codes_airport_array):
    weather_impact = 0

    for i in range(len(route) - 1):
        # Convert city name to ICAO code
        departure_icao = route[i][:4].upper()
        arrival_icao = route[i + 1][:4].upper()

        # Find indices of departure and arrival in the ICAO codes array
        departure_index = list(icao_codes_airport_array).index(departure_icao)
        arrival_index = list(icao_codes_airport_array).index(arrival_icao)

        # Get coordinates of departure and arrival airports
        departure_coords = (latitude_array[departure_index], longitude_array[departure_index])
        arrival_coords = (latitude_array[arrival_index], longitude_array[arrival_index])

        # Calculate flight direction
        flight_direction = calculate_bearing(*departure_coords, *arrival_coords)

        # Find weather data index
        weather_index = list(date_array).index(date_of_travel)
        wind_speed = wind_speed_array[weather_index]
        wind_direction = wind_direction_array[weather_index]

        # Determine if headwind or tailwind
        wind_angle = abs(wind_direction - flight_direction) % 360
        if wind_angle > 180:
            wind_angle = 360 - wind_angle

        # Headwind if angle < 90, tailwind if angle > 90
        if wind_angle <= 90:
            # Headwind: Increases fuel consumption and time
            weather_impact += wind_speed * 0.1  # Example impact calculation
        else:
            # Tailwind: Decreases fuel consumption and time
            weather_impact -= wind_speed * 0.1  # Example impact calculation

    # Normalize the weather impact
    normalized_impact = 1 / (1 + abs(weather_impact))
    return normalized_impact



def total_distance_fitness(total_distance):
    # Shorter distance is better
    return 1 / total_distance

def main_fitness_function(route, aircraft, date_of_travel, aircraft_data, weather_data, airport_data):
    # Calculate sub-fitness values
    fuel_speed_fitness = fuel_consumption_speed_fitness(route, aircraft, cruise_speed_array, fuel_consumption_array, latitude_array, longitude_array, icao_codes_airport_array)
    weather_fitness = weather_impact_fitness(route, date_of_travel, weather_data, wind_speed_array, wind_direction_array, city_weather_array, date_array, latitude_array, longitude_array, icao_codes_airport_array)
    distance_fitness = total_distance_fitness(calculate_total_distance(route, latitude_array, longitude_array, icao_codes_airport_array))

    # Combine sub-fitness values to calculate final fitness
    # The combination can be a weighted sum or other method depending on the importance of each factor
    final_fitness = fuel_speed_fitness + weather_fitness + distance_fitness

    return final_fitness


def tournament_selection(population, tournament_size=3):
    selected_parents = []
    actual_tournament_size = min(tournament_size, len(population))  # Adjust tournament size if necessary

    for _ in range(len(population)):
        tournament = random.sample(population, actual_tournament_size)
        winner = max(tournament, key=lambda chromosome: chromosome['fitness'])
        selected_parents.append(winner)

    return selected_parents

def is_valid_offspring(route, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array):
    # Check for repeated cities in the route
    if len(route) != len(set(route)):
        return False

    # Find the index of the aircraft in the aircraft type array
    try:
        aircraft_index = list(aircraft_type_array).index(aircraft)
    except ValueError:
        return False  # Aircraft not found in the array

    # Get the maximum range for this aircraft
    aircraft_max_range = max_range_array[aircraft_index]

    # Check each leg of the route
    for i in range(len(route) - 1):
        departure_icao = route[i][:4].upper()
        arrival_icao = route[i + 1][:4].upper()

        # Ensure departure and arrival ICAO codes are in the airport dataset
        if departure_icao not in icao_codes_airport_array or arrival_icao not in icao_codes_airport_array:
            return False

        # Calculate distance for each leg
        departure_index = list(icao_codes_airport_array).index(departure_icao)
        arrival_index = list(icao_codes_airport_array).index(arrival_icao)
        departure_coords = (latitude_array[departure_index], longitude_array[departure_index])
        arrival_coords = (latitude_array[arrival_index], longitude_array[arrival_index])
        distance = haversine(departure_coords, arrival_coords)

        # Check if the distance is within the aircraft's range
        if distance > aircraft_max_range:
            return False

    return True


def crossover(parent1, parent2, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array, max_attempts=100):
    attempts = 0
    while attempts < max_attempts:
        # Choose a crossover point
        crossover_point = random.randint(1, min(len(parent1['route']), len(parent2['route'])) - 2)
        
        # Create new route by combining segments of parent routes
        offspring_route = parent1['route'][:crossover_point] + parent2['route'][crossover_point:]

        # Check if the offspring route is valid
        if is_valid_offspring(offspring_route, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array):
            return offspring_route

        attempts += 1

    return None  # Return None if no valid offspring is created within the maximum number of attempts




def mutate(route, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array, mutation_rate, max_attempts=100):
    attempts = 0
    while attempts < max_attempts:
        mutated_route = route.copy()
        if random.random() < mutation_rate:
            # Choose a random position for mutation
            mutation_point = random.randint(1, len(mutated_route) - 2)
            new_airport = random.choice(icao_codes_airport_array)
            mutated_route[mutation_point] = new_airport[:4].upper()

            # Check if the mutated route is valid
            if is_valid_offspring(mutated_route, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array):
                return mutated_route

        attempts += 1

    return route  # Return the original route if no valid mutation is created within the maximum number of attempts


def genetic_algorithm(departure_icao, destination_city, date_of_travel, num_generations, population_size, mutation_rate):
    # Initial Population
    population = create_initial_population(population_size, departure_icao, destination_city, aircraft_type_array, icao_codes_aircraft_array, max_range_array, latitude_array, longitude_array, icao_codes_airport_array, city_airport_array)
    #print(population)
    best_route = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        #print("1")
        # Calculate fitness for each chromosome
        for chromosome in population:
            chromosome['fitness'] = main_fitness_function(chromosome['route'], chromosome['aircraft'], date_of_travel, aircraft_data, weather_data, airport_data)

            # Update best route
            if chromosome['fitness'] > best_fitness:
                best_route = chromosome
                best_fitness = chromosome['fitness']

        # Selection
        parents = tournament_selection(population)

        # Crossover and Mutation
        offspring_population = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):  # Ensure there is a pair to crossover
                parent1 = parents[i]
                parent2 = parents[i+1]
                aircraft = parent1['aircraft']  # Assuming the aircraft is the same for both parents

                offspring_route = crossover(parent1, parent2, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array)

                if offspring_route is not None:
                    offspring_route = mutate(offspring_route, aircraft, aircraft_type_array, max_range_array, icao_codes_airport_array, latitude_array, longitude_array, mutation_rate)

                    offspring = {
                        'route': offspring_route,
                        # Recalculate aircraft, total_distance, and fitness for the offspring
                        'aircraft': aircraft,
                        'total_distance': calculate_total_distance(offspring_route, latitude_array, longitude_array, icao_codes_airport_array),
                    }
                    offspring['fitness'] = main_fitness_function(offspring_route, aircraft, date_of_travel, aircraft_data, weather_data, airport_data)
                    offspring_population.append(offspring)

        # Replace the old population with the new one
        population = offspring_population

        # Optional: Check for termination condition (like no improvement in best fitness)

    return {'route': best_route['route']} if best_route else {}

# Example usage
#num_generations = 100
#population_size = 20
#mutation_rate = 0.05
#optimal_route = genetic_algorithm(departure_icao, destination_city, date_of_travel, num_generations, population_size, mutation_rate)
#print(optimal_route)
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')  # Replace with your HTML file name


@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    num_generations = 100
    population_size = 20
    mutation_rate = 0.05
    
    departure_city = request.form['departureCity']
    destination_city = request.form['destinationCity']
    date_of_travel = request.form['dateInput']
    
    # User input for the departure airport's ICAO code
    departure_icao = departure_city[:4].upper()
    # User input for the destination city
    #destination_city = input("Enter the name of the destination city: ").title()

    # Call your genetic algorithm function here
    optimal_route = genetic_algorithm(departure_icao, destination_city, date_of_travel, num_generations, population_size, mutation_rate)
    
    return optimal_route

if __name__ == '__main__':
    app.run(debug=True, port=5012)  # Change port to 5001 or another unused port

