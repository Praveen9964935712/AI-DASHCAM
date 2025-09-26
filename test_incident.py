"""
test_incident.py
Test script for incident.py
"""
from incident import IncidentLogger

def main():
    logger = IncidentLogger()
    logger.log_incident('Test event', 'test_video.mp4')
    print('Incident logged.')

if __name__ == '__main__':
    main()
