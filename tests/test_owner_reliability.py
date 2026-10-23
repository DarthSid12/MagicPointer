import unittest
import numpy as np

from magicpointer.cameras import Frame
from magicpointer.engine import Engine, Gesture, Observation
from magicpointer.ownership import HandOwner
from test_ownership_fusion import pose


class RetainedControlTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(activation_dwell=.5, retain_activation_on_loss=True, activation_grace=.4)
        self.now = 0.

    def step(self, gesture, dt=.13, tip=(0., 0., 2.), pitch=None):
        self.now += dt
        obs = Observation(gesture, (0.,0.,2.), tip, pitch) if gesture else None
        return self.engine.step(obs, self.now)

    def activate(self):
        for _ in range(6):
            self.step(Gesture.FIST)
        self.assertTrue(self.engine.active)

    def test_low_fps_rejected_pair_pauses_then_recovers_without_reactivation_or_jump(self):
        self.activate()
        self.step(Gesture.POINT, tip=(.05,.02,2.))
        cursor = self.engine.cursor
        self.assertEqual(self.step(None, dt=.17), [])
        self.assertTrue(self.engine.paused)
        self.assertEqual(self.step(Gesture.POINT, dt=.17, tip=(.2,.1,2.)), [])
        self.assertTrue(self.engine.active)
        self.assertTrue(self.engine.paused)
        self.step(Gesture.POINT, tip=(.2,.1,2.))
        self.assertEqual(self.engine.cursor, cursor)
        self.assertFalse(self.engine.paused)

    def test_long_loss_retains_owner_and_never_reuses_pending_click(self):
        self.activate()
        self.step(Gesture.RIGHT)
        self.step(None, dt=1.)
        self.assertTrue(self.engine.active)
        self.assertEqual(self.step(Gesture.RIGHT), [])
        events=[]
        for _ in range(10): events += self.step(Gesture.RIGHT)
        self.assertFalse(any(event.kind=='click' for event in events))
        for _ in range(3): self.step(Gesture.POINT)
        events=[]
        for _ in range(3): events += self.step(Gesture.RIGHT)
        self.assertEqual([event.value for event in events if event.kind=='click'], ['right'])

    def test_brief_missing_pair_is_bridged_without_pause_or_dwell_credit(self):
        self.activate()
        for _ in range(3): self.step(Gesture.POINT, dt=.03)
        events = self.step(Gesture.LEFT, dt=.03)
        # Missing pairs within the grace period: no pause, no latch, no hold time.
        events += self.step(None, dt=.03) + self.step(None, dt=.03)
        self.assertFalse(self.engine.paused)
        events += self.step(Gesture.LEFT, dt=.03)
        self.assertFalse(any(event.kind == 'click' for event in events))
        for _ in range(5): events += self.step(Gesture.LEFT, dt=.03)
        self.assertEqual([event.value for event in events if event.kind == 'click'], ['left'])

    def test_brief_unknown_or_missing_does_not_discard_valid_fist_evidence(self):
        self.step(Gesture.FIST)
        self.step(Gesture.FIST)
        self.step(None)
        evidence=self.engine.activation_progress
        self.step(Gesture.FIST)
        self.assertEqual(self.engine.activation_progress, evidence)
        for _ in range(3): self.step(Gesture.FIST)
        self.assertTrue(self.engine.active)

    def test_known_other_gesture_identity_change_and_long_gap_reset_activation(self):
        for interruption in (Gesture.POINT, Gesture.PALM):
            self.engine.reset_activation()
            self.step(Gesture.FIST)
            self.step(Gesture.FIST)
            self.step(interruption)
            self.assertEqual(self.engine.activation_progress, 0.)
        self.step(Gesture.FIST)
        self.step(Gesture.FIST)
        self.engine.reset_activation()
        self.assertEqual(self.engine.activation_progress, 0.)
        self.step(Gesture.FIST)
        self.step(Gesture.FIST)
        self.step(None, dt=.5)
        self.assertEqual(self.engine.activation_progress, 0.)

    def test_scroll_rate_is_cleared_on_pause_and_new_neutral_required(self):
        self.activate()
        for _ in range(10): self.step(Gesture.SCROLL, pitch=0.)
        for _ in range(10): self.step(Gesture.SCROLL, pitch=30.)
        self.assertTrue(self.engine.scrolling)
        self.step(None)
        self.assertIsNone(self.engine.scroll.neutral)
        self.assertEqual(self.engine.scroll.speed, 0.)
        events=[]
        for _ in range(10): events += self.step(Gesture.SCROLL, pitch=30.)
        self.assertFalse(any(event.kind=='scroll' for event in events))


class BoundHandEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.owner=HandOwner(2)
        self.frames=[Frame(np.zeros((1080,1920,3),np.uint8),0.) for _ in range(2)]
        selected=self.owner.select([[pose()],[pose(x=600.)]],self.frames,1.)
        self.owner.lock(selected.key)
        self.key=selected.key

    def test_nearby_owner_label_flip_preserves_ids_and_fast_two_view_palm_releases(self):
        left,right=pose(Gesture.PALM),pose(Gesture.PALM,x=600.)
        right.label='Left'
        result=self.owner.select([[left],[right]],self.frames,1.13)
        self.assertEqual(result.key,self.key)
        self.assertEqual(result.combined.gesture,Gesture.PALM)
        self.assertTrue(self.owner.release_ready)
        self.assertEqual(self.owner.diagnostics['palm_release_evidence'],'two_view_agreement')
        self.assertTrue(any(cost['allowed_owner_label_flip'] for cost in self.owner.locals[1].diagnostics['costs']))

    def test_far_different_label_hand_cannot_replace_owner(self):
        result=self.owner.select([[pose()],[pose(x=1500.,label='Left')]],self.frames,1.13)
        self.assertIsNone(result.combined)
        self.assertEqual(self.owner.owner,self.key)
        self.assertFalse(self.owner.release_ready)

    def test_single_or_disagreeing_palm_estimate_does_not_get_fast_release(self):
        left,right=pose(Gesture.PALM),pose(Gesture.PALM,x=600.)
        right.gesture=Gesture.UNKNOWN
        self.owner.select([[left],[right]],self.frames,1.13)
        self.assertFalse(self.owner.release_ready)
        self.owner.select([[left],[right]],self.frames,1.39)
        self.assertTrue(self.owner.release_ready)


    def test_single_camera_palm_still_requires_confirmation_hold(self):
        owner=HandOwner(1)
        frames=self.frames[:1]
        selected=owner.select([[pose()]],frames,1.)
        owner.lock(selected.key)
        owner.select([[pose(Gesture.PALM)]],frames,1.13)
        self.assertFalse(owner.release_ready)
        owner.select([[pose(Gesture.PALM)]],frames,1.39)
        self.assertTrue(owner.release_ready)
