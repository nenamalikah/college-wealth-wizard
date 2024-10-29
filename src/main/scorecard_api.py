import requests

# Define the base URL for the College Scoreboard API
BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools.json"

# Define your API key (replace 'YOUR_API_KEY' with your actual API key)
API_KEY = "CghFJ6dmrniJJlwsIHvhjIcOA7Qe82VPIMbwGkoj"

# Set up parameters for the API request
params = {
    'api_key': API_KEY,
    'school.name':'George Washington University',
    'fields': 'id, name, location,latest.cost.tuition,latest.admissions.admission_rate',
    'per_page': 10,  # Number of results per page
    'page': 0        # Page number for pagination
}

# Make the GET request to the API
response = requests.get(BASE_URL, params=params)

print(response.json())

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # Print the results
    for college in data['results']:
        print(f"ID: {college['id']}, Name: {college['name']}, "
              f"Location: {college['location']['city']}, "
              f"Tuition: {college['latest']['cost']['tuition']}, "
              f"Admission Rate: {college['latest']['admissions']['admission_rate']}")
else:
    print(f"Error: {response.status_code}, {response.text}")
