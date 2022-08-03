import unittest
from time import sleep, time

import pytest

from islearn.helpers import parallel_all, parallel_any


class TestHelpers(unittest.TestCase):
    @pytest.mark.skip("Doesn't work in CI")
    def test_parallel_all(self):
        def pred(n: int) -> bool:
            sleep(.005)
            return n > 0

        positive_numbers = range(1, 1000)

        start = time()
        self.assertTrue(all(pred(n) for n in positive_numbers))
        all_time = time() - start

        start = time()
        self.assertTrue(parallel_all(pred, positive_numbers))
        parallel_all_time = time() - start

        self.assertGreater(all_time, 8 * parallel_all_time)

        one_wrong = (-1 if n == 800 else n for n in range(1, 1000))
        start = time()
        self.assertFalse(all(pred(n) for n in one_wrong))
        all_time = time() - start

        one_wrong = (-1 if n == 800 else n for n in range(1, 1000))
        start = time()
        self.assertFalse(parallel_all(pred, one_wrong))
        parallel_all_time = time() - start

        self.assertGreater(all_time, 8 * parallel_all_time)

    @pytest.mark.skip("Doesn't work in CI")
    def test_parallel_any(self):
        def pred(n: int) -> bool:
            sleep(.005)
            return n < 0

        positive_numbers = range(1, 1000)

        start = time()
        self.assertFalse(any(pred(n) for n in positive_numbers))
        any_time = time() - start

        start = time()
        self.assertFalse(parallel_any(pred, positive_numbers))
        parallel_any_time = time() - start

        self.assertGreater(any_time, 8 * parallel_any_time)

        one_wrong = (-1 if n == 800 else n for n in range(1, 1000))
        start = time()
        self.assertTrue(any(pred(n) for n in one_wrong))
        any_time = time() - start

        one_wrong = (-1 if n == 800 else n for n in range(1, 1000))
        start = time()
        self.assertTrue(parallel_any(pred, one_wrong))
        parallel_any_time = time() - start

        self.assertGreater(any_time, 8 * parallel_any_time)


if __name__ == '__main__':
    unittest.main()
