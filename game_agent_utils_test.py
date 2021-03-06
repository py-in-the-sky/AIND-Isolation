import game_agent_utils.symmetry as sym

import unittest
from random import randint, randrange


class SymmetryTest(unittest.TestCase):
    def test_reflections_and_rotations(self):
        "Property tests of the reflection and rotation functions."
        for _ in range(250):
            w, h = randint(1, 15), randint(1, 15)  # Rectangular grid.
            r, c = randrange(0, h), randrange(0, w)
            loc = (r, c)

            # Testing that complementary actions undo each other.
            self.assertEqual(loc, sym.reflect_vertical(sym.reflect_vertical(loc, w, h), w, h))
            self.assertEqual(loc, sym.reflect_horizontal(sym.reflect_horizontal(loc, w, h), w, h))
            self.assertEqual(loc, sym.rotate_180(sym.rotate_180(loc, w, h), w, h))

        for _ in range(250):
            h = w = randint(1, 15)  # Square grid.
            r, c = randrange(0, h), randrange(0, w)
            loc = (r, c)

            # Testing that complementary actions undo each other.
            self.assertEqual(loc, sym.reflect_vertical(sym.reflect_vertical(loc, w, h), w, h))
            self.assertEqual(loc, sym.reflect_horizontal(sym.reflect_horizontal(loc, w, h), w, h))
            self.assertEqual(loc, sym.reflect_secondary_diagonal(sym.reflect_secondary_diagonal(loc, w, h), w, h))
            self.assertEqual(loc, sym.reflect_primary_diagonal(sym.reflect_primary_diagonal(loc, w, h), w, h))
            self.assertEqual(loc, sym.rotate_270(sym.rotate_90(loc, w, h), w, h))
            self.assertEqual(loc, sym.rotate_180(sym.rotate_180(loc, w, h), w, h))
            self.assertEqual(loc, sym.rotate_270(sym.rotate_90(loc, w, h), w, h))

            # Tesing equivalences -- e.g., that rotating 90 degrees and then another 90 degress
            # is the same as rotating 180 degrees in one go.
            self.assertEqual(sym.rotate_180(loc, w, h), sym.rotate_90(
                                                          sym.rotate_90(loc, w, h), w, h))
            self.assertEqual(sym.rotate_270(loc, w, h), sym.rotate_180(
                                                          sym.rotate_90(loc, w, h), w, h))
            self.assertEqual(sym.rotate_270(loc, w, h), sym.rotate_90(
                                                          sym.rotate_180(loc, w, h), w, h))
            self.assertEqual(sym.rotate_270(loc, w, h), sym.rotate_90(
                                                          sym.rotate_90(
                                                            sym.rotate_90(loc, w, h), w, h), w, h))
            self.assertEqual(sym.rotate_180(loc, w, h), sym.reflect_vertical(
                                                          sym.reflect_horizontal(loc, w, h), w, h))
            self.assertEqual(sym.rotate_90(loc, w, h), sym.reflect_vertical(
                                                         sym.reflect_horizontal(
                                                           sym.reflect_vertical(
                                                             sym.reflect_secondary_diagonal(loc, w, h), w, h), w, h), w, h))


if __name__ == '__main__':
    unittest.main()
