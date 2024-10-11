# Flight-Route-Optimization-using-Genetic-Algorithm

This project optimizes flight routes using Genetic Algorithms, focusing on reducing fuel consumption, enhancing efficiency, and adapting to real-time conditions.

Project Overview

Traditional route planning methods often fail to account for dynamic factors like weather and air traffic. This project addresses these challenges by developing a robust system to optimize flight routes.

Key Features

	•	Data Processing: Loads and processes aircraft, airport, and weather data.
	•	Route Optimization: Implements a Genetic Algorithm (GA) to find optimal routes considering distance, weather, and fuel consumption.
	•	Fitness Function: Evaluates routes based on factors like total distance, weather impact, and fuel use.
	•	Genetic Operations: Uses tournament selection, crossover, and mutation to evolve route solutions.
	•	User Interface: A Flask web app allows users to input travel details and receive optimized routes.

Usage

	1.	Clone the repository.
	2.	Install dependencies: pip install -r requirements.txt.
	3.	Run the Flask app: python app.py.
	4.	Access the interface at http://localhost:5000.

Future Enhancements

	•	Integrate real-time weather data.
	•	Refine mutation and crossover strategies for better optimization.

Contributors

	•	Moonis Haider
