"""
tests.py: Unit tests for Edge AI Dashcam core modules
"""
from models import Detection, Track, RiskEvent

def test_detection():
    d = Detection(0,0,10,10,0.9,2)
    assert d.x1 == 0 and d.class_id == 2
    print('Detection test passed')

def test_track():
    t = Track(1, [0,0,10,10], 0.8, 2)
    assert t.track_id == 1
    print('Track test passed')

def test_risk_event():
    r = RiskEvent('tailgating', 'high', 1, 1.2, 1234567890.0)
    assert r.event_type == 'tailgating'
    print('RiskEvent test passed')

if __name__ == '__main__':
    test_detection()
    test_track()
    test_risk_event()
    print('All tests passed!')
