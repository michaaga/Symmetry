import Symmetry    # The code to test
import unittest   # The test framework
import testSymmetry

class Test_TestSymmetry(unittest.TestCase):
    def test_reflection(self):
        self.assertTrue(testSymmetry.testReflectPoint())

    def test_Normalization(self):
        self.assertTrue(testSymmetry.testNormalizeLandmarks())


if __name__ == '__main__':
    unittest.main()