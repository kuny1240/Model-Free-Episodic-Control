import numpy as np
from pyflann import FLANN

class DND:
  def __init__(self, kernel, num_neighbors, max_memory, lr):
    self.kernel = kernel
    self.num_neighbors = num_neighbors
    self.max_memory = max_memory
    self.lr = lr
    self.keys = None
    self.values = None
    self.kdtree = FLANN()

    # key_cache stores a cache of all keys that exist in the DND
    # This makes DND updates efficient
    self.key_cache = {}
    # stale_index is a flag that indicates whether or not the index in self.kdtree is stale
    # This allows us to only rebuild the kdtree index when necessary
    self.stale_index = True
    # indexes_to_be_updated is the set of indexes to be updated on a call to update_params
    # This allows us to rebuild only the keys of key_cache that need to be rebuilt when necessary
    self.indexes_to_be_updated = set()

    # Keys and value to be inserted into self.keys and self.values when commit_insert is called
    self.keys_to_be_inserted = None
    self.values_to_be_inserted = None

    # Move recently used lookup indexes
    # These should be moved to the back of self.keys and self.values to get LRU property
    self.move_to_back = set()

  def get_index(self, key):
    """
    If key exists in the DND, return its index
    Otherwise, return None
    """
    if self.key_cache.get(key) is not None:
      if self.stale_index:
        self.commit_insert()
      return int(self.kdtree.nn_index(key, 1)[0][0])
    else:
      return None

  def update(self, value, index):
    """
    Set self.values[index] = value
    """
    values = self.values.data
    values[index] = value[0].data
    self.values = values

  def insert(self, key, value):
    """
    Insert key, value pair into DND
    """
    if self.keys_to_be_inserted is None:
      # Initial insert
      self.keys_to_be_inserted = key
      self.values_to_be_inserted = value
    else:
      self.keys_to_be_inserted = np.vstack([self.keys_to_be_inserted, key])
      self.values_to_be_inserted = np.vstack([self.values_to_be_inserted, value.data])
    self.key_cache[key] = 0
    self.stale_index = True

  def commit_insert(self):
    if self.keys is None:
      self.keys = self.keys_to_be_inserted
      self.values = self.values_to_be_inserted
    elif self.keys_to_be_inserted is not None:
      self.keys = np.vstack([self.keys, self.keys_to_be_inserted])
      self.values = np.vstack([self.values, self.values_to_be_inserted])

    # Move most recently used key-value pairs to the back
    if len(self.move_to_back) != 0:
      self.keys = np.vstack([self.keys[list(set(range(len(
          self.keys))) - self.move_to_back)], self.keys[list(self.move_to_back)]])
      self.values = np.vstack([self.values[list(set(range(len(
          self.values))) - self.move_to_back)], self.values[list(self.move_to_back)]])
      self.move_to_back = set()

    if len(self.keys) > self.max_memory:
      # Expel oldest key to maintain total memory
      for key in self.keys[:-self.max_memory]:
        del self.key_cache[key]
      self.keys = self.keys[-self.max_memory:]
      self.values = self.values[-self.max_memory:]
    self.keys_to_be_inserted = None
    self.values_to_be_inserted = None

    self.kdtree.build_index(self.keys)
    self.stale_index = False

  def lookup(self, lookup_key, update_flag=False):
    """
    Perform DND lookup
    If update_flag == True, add the nearest neighbor indexes to self.indexes_to_be_updated
    """
    lookup_indexes = self.kdtree.nn_index(
        lookup_key, min(self.num_neighbors, len(self.keys)))[0][0]
    output = 0
    kernel_sum = 0
    for i, index in enumerate(lookup_indexes):
      if i == 0 and self.key_cache.get(lookup_key[0]) is not None:
        # If a key exactly equal to lookup_key is used in the DND lookup calculation
        # then the loss becomes non-differentiable. Just skip this case to avoid the issue.
        continue
      if update_flag:
        self.indexes_to_be_updated.add(int(index))
      else:
        self.move_to_back.add(int(index))
      kernel_val = self.kernel(self.keys[int(index)], lookup_key[0])
      output += kernel_val * self.values[int(index)]
      kernel_sum += kernel_val
    output = output / kernel_sum
    return output

  def update_params(self):
    """
    Update self.keys and self.values via backprop
    Use self.indexes_to_be_updated to update self.key_cache accordingly and rebuild the index of self.kdtree
    """
    for index in self.indexes_to_be_updated:
      del self.key_cache[self.keys[index]]
    for index in self.indexes_to_be_updated:
      self.key_cache[self.keys[index]] = 0
    self.indexes_to_be_updated = set()
    self.kdtree.build_index(self.keys)
    self.stale_index = False