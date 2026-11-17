import sys
import unittest
from unittest.mock import patch

from config import Config
from main import parse_args, update_config_from_args


class ConfigSeedTests(unittest.TestCase):
    def test_cli_seed_is_preserved_in_config(self):
        with patch.object(sys, "argv", ["main.py", "--seed", "123"]):
            args = parse_args()

        config = update_config_from_args(Config(), args)

        self.assertEqual(config.seed, 123)


if __name__ == "__main__":
    unittest.main()
