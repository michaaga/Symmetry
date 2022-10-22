import unittest   # The test framework
import test_Symmetry

class Test_TestSymmetry(unittest.TestCase):
    def test_reflection(self):
        self.assertTrue(test_Symmetry.testReflectPoint())

    def test_Normalization(self):
        self.assertTrue(test_Symmetry.testNormalizeLandmarks())

    def test_HorizontalSymmetryOfLine(self):
        self.assertTrue(test_Symmetry.testHorizontalSymmetryOfLine())

    def test_NormalizeLandmarks(self):
        self.assertTrue(test_Symmetry.testNormalizeLandmarks())

if __name__ == '__main__':
    unittest.main()