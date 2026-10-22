import math
import unittest

import numpy as np

from magicpointer.cameras import Frame
from magicpointer.engine import Gesture
from magicpointer.fusion import combine
from magicpointer.ownership import HandOwner, LocalHands
from magicpointer.tracking import Hand, classify
from test_tracking import hand


def pose(gesture=Gesture.FIST, x=400., label='Right', rotation=0.):
    thumb, fingers = {Gesture.FIST: (False, [0, 0, 0, 0]), Gesture.POINT: (False, [1, 0, 0, 0]),
                      Gesture.PALM: (True, [1, 1, 1, 1]), Gesture.LEFT: (True, [1, 0, 0, 0]),
                      Gesture.RIGHT: (True, [1, 1, 0, 0]), Gesture.SCROLL: (False, [1, 1, 0, 0])}[gesture]
    world = hand(thumb, fingers)
    angle = math.radians(rotation)
    matrix = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    world = world @ matrix.T
    pixels = hand(thumb, fingers)[:, :2]*700+[x, 550.]
    return Hand(pixels, world, label, classify(world))


class FusionTests(unittest.TestCase):
    def test_all_gestures_fuse_across_camera_rotation_scale_translation(self):
        for gesture in (Gesture.FIST, Gesture.POINT, Gesture.LEFT, Gesture.RIGHT, Gesture.SCROLL, Gesture.PALM):
            left, right = pose(gesture), pose(gesture, rotation=50.)
            right.world = np.asarray(right.world)*1.7+[2., -3., 1.]
            fused, _, spread = combine([left, right])
            self.assertIsNotNone(fused)
            self.assertEqual(fused.gesture, gesture)
            self.assertLess(spread, 1e-10)

    def test_shared_classification_uses_landmarks_instead_of_independent_labels(self):
        left, right = pose(Gesture.POINT), pose(Gesture.POINT, rotation=-30.)
        left.gesture, right.gesture = Gesture.FIST, Gesture.PALM
        fused, _, _ = combine([left, right])
        self.assertEqual(fused.gesture, Gesture.POINT)
        right.world = pose(Gesture.PALM).world
        fused, _, _ = combine([left, right])
        self.assertTrue(fused is None or fused.gesture != Gesture.POINT)

    def test_missing_or_invalid_view_contributes_no_geometry(self):
        left, right = pose(), pose(label='Left')
        self.assertEqual(combine([left, None])[0].gesture, Gesture.FIST)
        self.assertEqual(combine([None, left])[0].gesture, Gesture.FIST)
        self.assertIsNone(combine([left, right])[0])
        right.label, right.world = 'Right', []
        self.assertEqual(combine([left, right])[0].gesture, Gesture.FIST)
        self.assertIsNone(combine([None, right])[0])

    def test_bound_views_combine_geometry_despite_disagreeing_pose_estimates(self):
        from magicpointer.fusion import canonical
        left, right = pose(Gesture.POINT), pose(Gesture.PALM)
        details = {}
        fused, _, spread = combine([left, right], bound_pair=True, diagnostics=details)
        self.assertGreater(spread, .4)
        np.testing.assert_allclose(fused.world, (canonical(left)+canonical(right))/2)
        self.assertEqual(fused.gesture, classify(fused.world))
        self.assertEqual(details['evidence_views'], 2)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.owner = HandOwner(2)
        image = np.zeros((1080, 1920, 3), np.uint8)
        self.frames = [Frame(image, 0.), Frame(image, .005)]
        self.now = 0.

    def step(self, left, right, dt=.03):
        self.now += dt
        return self.owner.select([left, right], self.frames, self.now)

    def test_detection_order_reversal_keeps_local_identity(self):
        tracks = LocalHands()
        first, second = pose(x=300.), pose(Gesture.POINT, x=1000.)
        visible = tracks.update([first, second], 0., (1920, 1080))
        first_id = next(key for key, value in visible.items() if value is first)
        second_id = next(key for key, value in visible.items() if value is second)
        visible = tracks.update([second, first], .03, (1920, 1080))
        self.assertIs(visible[first_id], first)
        self.assertIs(visible[second_id], second)

    def test_crossing_ambiguous_hands_pause_instead_of_swapping(self):
        tracks = LocalHands()
        tracks.update([pose(x=400.), pose(x=500.)], 0., (1920, 1080))
        self.assertEqual(tracks.update([pose(x=449.), pose(x=451.)], .03, (1920, 1080)), {})

    def test_select_one_fist_with_other_hand_visible_and_lock_survives_order_change(self):
        own = [pose(x=350.), pose(x=500.)]
        other = [pose(Gesture.POINT, x=1100., label='Left'), pose(Gesture.POINT, x=1250., label='Left')]
        selected = self.step([own[0], other[0]], [other[1], own[1]])
        self.assertEqual(selected.combined.gesture, Gesture.FIST)
        self.assertEqual(selected.progress, 0.)
        key = selected.key
        for _ in range(90):
            selected = self.step([other[0], own[0]], [own[1], other[1]])
        self.assertEqual(selected.key, key)
        self.assertEqual(selected.progress, 1.)
        self.owner.lock(key)
        # The other hand may open its palm; only the owner's gesture counts.
        selected = self.step([pose(Gesture.PALM, 1100., 'Left'), pose(Gesture.POINT, 350.)],
                             [pose(Gesture.POINT, 500.), pose(Gesture.PALM, 1250., 'Left')])
        self.assertEqual(selected.key, key)
        self.assertEqual(selected.combined.gesture, Gesture.POINT)
        self.assertFalse(self.owner.release_ready)

    def test_same_handed_same_shape_pair_ambiguity_blocks_selection(self):
        selected = self.step([pose(x=300.), pose(x=1100.)], [pose(x=400.), pose(x=1300.)])
        self.assertIsNone(selected.combined)
        self.assertIsNone(self.owner.owner)

    def test_owner_missing_from_one_view_keeps_lock_without_other_hand_takeover(self):
        selected = self.step([pose()], [pose(x=600.)])
        self.owner.lock(selected.key)
        key = selected.key
        selected = self.step([pose(), pose(Gesture.PALM, 1200., 'Left')], [pose(Gesture.PALM, 1300., 'Left')])
        self.assertEqual(selected.combined.gesture, Gesture.FIST)
        self.assertIsNone(selected.hands[1])
        self.assertEqual(self.owner.owner, key)
        self.assertFalse(self.owner.release_ready)
        selected = self.step([pose()], [pose(x=600.)])
        self.assertEqual(selected.key, key)
        self.assertIsNotNone(selected.combined)

    def test_single_view_selection_then_unique_second_view_binding(self):
        selected = self.step([], [pose(x=600.)])
        self.assertEqual(selected.combined.gesture, Gesture.FIST)
        self.assertIsNone(selected.key[0])
        self.owner.lock(selected.key)
        selected = self.step([pose(label='Left')], [pose(x=600.)])
        self.assertIsNone(selected.key[0])  # unrelated hand cannot supply perspective
        selected = self.step([pose()], [pose(x=600.)])
        self.assertTrue(all(track is not None for track in selected.key))
        self.assertTrue(all(hand is not None for hand in selected.hands))

    def test_no_takeover_when_only_unbound_view_sees_a_hand(self):
        selected = self.step([pose()], [])
        self.owner.lock(selected.key)
        selected = self.step([], [pose(x=600.)])
        self.assertIsNone(selected.combined)
        self.assertIsNone(self.owner.owner[1])

    def test_unrelated_opposite_view_does_not_block_single_view_fist_selection(self):
        selected = self.step([pose()], [pose(Gesture.POINT, x=600., label='Left')])
        self.assertEqual(selected.combined.gesture, Gesture.FIST)
        self.assertIsNone(selected.key[1])

    def test_calibrated_new_perspective_can_join_despite_shape_disagreement(self):
        from magicpointer.stereo import Stereo
        from test_geometry import rig, project
        stereo = Stereo(rig())
        left, right = pose(), pose(Gesture.PALM)
        left.pixels, right.pixels = project(stereo, np.asarray(left.world)*.06+[0., 0., 2.])
        selected = self.owner.select([[left], []], self.frames, 1., stereo=stereo)
        self.owner.lock(selected.key)
        selected = self.owner.select([[left], [right]], self.frames, 1.1, stereo=stereo)
        self.assertTrue(all(track is not None for track in selected.key))
        self.assertIsNotNone(selected.combined)
        self.assertGreater(selected.spread, .4)

    def test_palm_release_is_available_even_if_engine_has_paused_for_tracking_loss(self):
        selected = self.step([pose()], [pose(x=600.)])
        self.owner.lock(selected.key)
        for _ in range(10):
            self.step([pose(Gesture.PALM)], [pose(Gesture.PALM, x=600.)])
        self.assertTrue(self.owner.release_ready)
        self.owner.release()
        self.assertIsNone(self.owner.owner)
        self.assertFalse(self.owner.release_ready)
