
# Remember to enable the computer vision mode in the settings.json file.

# AirSim packages.
import setup_path 
import airsim

# The AirSim client.
client = airsim.VehicleClient()

success1 = client.simSetSegmentationObjectID("[\w]*", 0, True)
success2 = client.simSetSegmentationObjectID("FlowerPot_319", 10, True)

print(f'success1 = {success1}, \nsuccess2 = {success2}')
