import unittest

class TestMyModuleImport(unittest.TestCase):

    def test_import(self):
        try:
            from vitalwave import basic_algos
        except ImportError:
            self.fail("Failed to import basic_algos")

if __name__ == '__main__':
    unittest.main()