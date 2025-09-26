"""
test_risk.py
Test script for risk.py
"""
from risk import RiskScorer

def main():
    scorer = RiskScorer()
    tracked_objects = [ [100, 100, 200, 200, 0.9, 0, 1], [300, 120, 400, 220, 0.8, 1, 2] ]
    ttc = scorer.compute_ttc(tracked_objects)
    headway = scorer.compute_headway(tracked_objects)
    lane_dev = scorer.lane_deviation(None, tracked_objects)
    print('TTC:', ttc)
    print('Headway:', headway)
    print('Lane deviation:', lane_dev)

if __name__ == '__main__':
    main()
