import sys, os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.utils import config
import copy

class testConfig(unittest.TestCase):
    def setUp(self):
        self.config_dir = os.path.join(os.getcwd(), "tests", "configs", "config_test")
        self.conf = config(self.config_dir)
        self.readable = self.conf.get_readable()
        self.ini_hash = self.conf.get_hash()
        
    def test_get_details(self):
        self.readable = self.conf.get_readable()
        self.ini_hash = self.conf.get_hash()

    def test_change(self):
        conf_old = copy.deepcopy(self.conf)
        self.conf.update("trainer.model.type", "SLMT", merge=False)
        self.conf.update("decoder.model", None, merge=False)
        new_hash = self.conf.get_hash()

        self.assertTrue(self.conf.get_value("trainer.model.type") == "SLMT", f"Modified Config Expected 'SLMT' in trainer.model.type, got {self.conf.get_value('trainer.model.type')}")
        self.assertTrue(self.conf.get_value("decoder.model") is None, f"Modified Config Expected None in decoder.model, got {self.conf.get_value('decoder.model')}")
        self.assertTrue(self.conf.has_changes(), "Expected changes to be detected")
        self.assertTrue(self.conf.has_changes(conf_old), "Expected changes to be detected compared to old config")
        self.assertTrue(self.readable != self.conf.get_readable(), "Expected readable config to change after modification")
        self.assertTrue(new_hash == self.ini_hash, f"Expected the same hash {self.ini_hash}, got {new_hash}")
        self.assertTrue(self.conf.get_changes() == {'decoder.model': ({'type': 'LSTM', 'params': {'input_size': 96, 'num_outputs': 4, 'hidden_size': 300, 'num_layers': 1, 'rnn_type': 'lstm', 'device': 'cuda', 'hidden_noise_std': 0.0, 'dropout_input': False, 'drop_prob': 0.0, 'sequence_length': 20}}, None), 'trainer.model.type': ('LSTM', 'SLMT')}, f"Unexpected changes detected: {self.conf.get_changes()}")

    def test_revert(self):
        self.conf.reset()
        self.assertTrue(self.conf.get_hash() == self.ini_hash, f"Expected hash {self.ini_hash}, got {self.conf.get_hash()}")
        self.assertFalse(self.conf.has_changes(), "Expected no changes after revert")

    def test_history(self):
        self.conf.reset()
        self.conf.update("trainer.model.type", "SLMT", merge=False)
        self.conf.update("decoder.model", None, merge=False)
        history = self.conf.get_history()
        self.assertTrue(len(history) == 2, f"Expected history length 2, got {len(history)}")
        self.assertTrue(history.iloc[0]['entry'] == 'trainer.model.type', f"Expected entry 'trainer.model.type' at row 0, got {history.iloc[0]['entry']}")
        self.assertTrue(history.iloc[0]['operation'] == {'value': 'SLMT', 'merge': False}, f"Expected operation {{'value': 'SLMT', 'merge': False}} at row 0, got {history.iloc[0]['operation']}")
        self.assertTrue(history.iloc[1]['entry'] == 'decoder.model', f"Expected entry 'decoder.model' at row 1, got {history.iloc[1]['entry']}")
        self.assertTrue(history.iloc[1]['operation'] == {'value': None, 'merge': False}, f"Expected operation {{'value': None, 'merge': False}} at row 1, got {history.iloc[1]['operation']}")

    def test_save_and_load(self):
        self.conf.reset()
        self.conf.update("trainer.model.type", "SLMT", merge=False)
        self.conf.update("decoder.model", None, merge=False)
        self.conf.save(file_name="SLMT_nodecode.yaml")

        saved_file_path = os.path.join(self.config_dir, "SLMT_nodecode.yaml")
        self.assertTrue(os.path.exists(saved_file_path), f"Expected saved config file to exist at {saved_file_path}")
        
        conf2 = config(os.path.join(self.config_dir, "SLMT_nodecode.yaml"))
        self.assertFalse(self.conf.has_changes(conf2), "Expected no changes between original and loaded config")

        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
            
if __name__ == "__main__":
    unittest.main()
