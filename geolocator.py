from geopy.geocoders import Nominatim
  
# initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")
  
  
# Latitude & Longitude input
Latitude = "11.0088682"
Longitude = "76.9595363"
  
location = geolocator.reverse(Latitude+","+Longitude)
  
address = location.raw['address']
  
# traverse the data
city = address.get('city', '')
state = address.get('state', '')
country = address.get('country', '')
code = address.get('country_code')
zipcode = address.get('postcode')
print('City : ', city)
print('State : ', state)
print('Country : ', country)
print('Zip Code : ', zipcode)