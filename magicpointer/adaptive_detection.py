"""Opt-in, two-camera debug reacquisition; never replaces cross-view matching."""
from .tracking import Tracker


class AdaptiveDetection:
    interval = .25
    confidence = .5

    def __init__(self, model, num_hands):
        self.model, self.num_hands = model, num_hands
        self.trackers = [None, None]
        self.next_attempt = [0., 0.]

    def apply(self, detections, frames, now, *, paired=True):
        result = list(detections)
        trace = {'baseline_counts': [len(hands) for hands in detections],
                 'attempted_view': None, 'candidate_count': 0,
                 'detection_source': ['regular', 'regular']}
        if not paired:
            trace['reason'] = 'pair timing rejected'
            return result, trace
        present = [bool(hands) for hands in detections]
        if present[0] == present[1]:
            trace['reason'] = 'both views detected' if present[0] else 'neither view detected'
            return result, trace
        missing = 1 if present[0] else 0
        if now < self.next_attempt[missing]:
            trace['reason'] = 'retry cooldown'
            return result, trace
        self.next_attempt[missing] = now + self.interval
        if self.trackers[missing] is None:
            # IMAGE mode performs fresh detection on each retry: no intermittent
            # low-threshold VIDEO track can silently persist between attempts.
            self.trackers[missing] = Tracker(self.model, num_hands=self.num_hands,
                                             detection_confidence=self.confidence,
                                             presence_confidence=self.confidence, image_mode=True)
        frame = frames[missing]
        recovered = self.trackers[missing].detect_all(frame.image, frame.grabbed_at)
        result[missing] = recovered
        trace.update(attempted_view=missing, candidate_count=len(recovered),
                     reason='recovery candidates' if recovered else 'retry found no hand')
        if recovered:
            trace['detection_source'][missing] = 'recovery'
        return result, trace

    def close(self):
        for tracker in self.trackers:
            if tracker is not None:
                tracker.close()
