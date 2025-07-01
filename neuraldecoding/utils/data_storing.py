import os
import pickle  # or any other serialization library

class DataManager:
    def __init__(self, basepath):
        self.basepath = basepath
        self.cache = {}

    def get_path(self, key):
        """Constructs the full path for a given key."""
        return os.path.join(self.basepath, f"{key}.pkl")  


    def load(self, key):
        """Loads the value from file if it exists, otherwise returns None."""
        path = self.get_path(key)
        
        if key in self.cache:
            return self.cache[key]
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                value = pickle.load(f)
                self.cache[key] = value  
            
            return value
            
        else:
            return None

    def save(self, key, value):
        """Saves the value to file and updates the cache."""
        path = self.get_path(key)
        with open(path, 'wb') as f:
            pickle.dump(value, f)
        self.cache[key] = value


# class DataManager:
#     def __init__(self, basepath):
#         self.basepath = basepath
#         self.cache = {}

#     def get_path(self, key):
#         """Constructs the full path for a given key."""
#         return os.path.join(self.basepath, f"{key}.pkl")  


#     def load(self, key):
#         """Loads the value from file if it exists, otherwise returns None."""
#         path = self.get_path(key)
        
#         if key in self.cache:
#             return self.cache[key]
        
#         if os.path.exists(path):
#             with open(path, 'rb') as f:
#                 value = pickle.load(f)
#                 self.cache[key] = value  
            
#         #     return value
            
#         # else:
#         #     return None
        
#         self.memoization(key_list)
        
        
        
#     @staticmethod
#     def memoization(keys):
        
        
        

#     def save(self, key, value):
#         """Saves the value to file and updates the cache."""
#         path = self.get_path(key)
#         with open(path, 'wb') as f:
#             pickle.dump(value, f)
#         self.cache[key] = value
