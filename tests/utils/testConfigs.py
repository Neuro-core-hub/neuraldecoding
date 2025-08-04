from neuraldecoding.utils import config
import copy
conf = config("D:/ND/github/neuraldecoding/tests/configs/config_test")

# 1. Print out Config Details and Hash
print("========================Config Details========================")
print(conf.get_readable())
print(conf.get_hash())

# 2. Change a value in the config
print("========================Change a Value========================")
conf_old = copy.deepcopy(conf)
conf.update_value("trainer.model.type", "SLMT", merge=False)
conf.update_value("decoder.model", None, merge=False)
print(conf.get_readable())
print(conf.get_hash())

print(conf.has_changes())
print(conf.has_changes(conf_old))
print(conf.get_changes())
print(conf.get_history())

# 3. Save the config
print("========================Save Config========================")
conf.save_config(file_name="SLMT_nodecode.yaml")

# 4. Reset to original config
print("========================Reset to Original Config========================")
conf.reset_to_original()
#print(conf.get_readable())
print(conf.get_hash())

print(conf.has_changes())

# 5. Load the config again
print("========================Load Config Again========================")
conf2 = config("D:/ND/github/neuraldecoding/tests/configs/config_test/SLMT_nodecode.yaml")
print(conf2.get_readable())
print(conf2.get_hash())