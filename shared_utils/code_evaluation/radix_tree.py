from typing import Optional
import pickle
from os import path

class RadixTreeNode:
    def __init__(self):
        self.children = {}
        self.value = None
        self.is_end_of_key = False

class RadixTree:
    SAVE_PATH = path.join(path.dirname(__file__), ".radix_tree_cache.bin")

    def __init__(self):
        self.root = RadixTreeNode()

    @classmethod
    def from_disk(cls, save_path: Optional[str] = None):
        cache_path = save_path or cls.SAVE_PATH
        print(cache_path)
        if path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                cached_tree = pickle.load(f)
            return cached_tree
        else:
            return cls()
        
    def save(self):
        with open(self.SAVE_PATH, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def add(self, key, value):
        """Adds a key-value pair to the trie. Raises an error if the key already exists."""
        current_node = self.root
        original_key = key
        while key:
            for prefix, child in current_node.children.items():
                common_prefix_length = self._longest_common_prefix_length(key, prefix)
                
                if common_prefix_length == len(prefix):  # Full match with an existing prefix
                    current_node = child
                    key = key[common_prefix_length:]
                    break
                
                elif common_prefix_length > 0:  # Partial match, split required
                    common_prefix = prefix[:common_prefix_length]
                    remaining_prefix = prefix[common_prefix_length:]
                    remaining_key = key[common_prefix_length:]
                    
                    # Split the current node
                    split_node = RadixTreeNode()
                    split_node.children[remaining_prefix] = child
                    
                    # Update the current node
                    current_node.children[common_prefix] = split_node
                    del current_node.children[prefix]
                    
                    # Continue processing with the new split node
                    current_node = split_node
                    key = remaining_key
                    break
            else:
                # No match found, insert a new node here
                new_node = RadixTreeNode()
                current_node.children[key] = new_node
                new_node.value = value
                new_node.is_end_of_key = True
                return

        # If we reach here, the key already exists
        if current_node.is_end_of_key:
            raise ValueError(f"Key '{original_key}' already exists.")
        
        # Mark the current node as the end of the key and set its value
        current_node.is_end_of_key = True
        current_node.value = value


    def contains(self, key):
        """Checks if the trie contains the given key."""
        node = self._find_node(key)
        return node.is_end_of_key if node else False

    def get(self, key):
        """Retrieves the value associated with the key, or None if the key does not exist."""
        node = self._find_node(key)
        return node.value if node and node.is_end_of_key else None

    def _find_node(self, key: str):
        """Finds the node corresponding to the key or returns None."""
        current_node = self.root
        while key:
            for prefix, child in current_node.children.items():
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    current_node = child
                    break
            else:
                return None
        return current_node

    @staticmethod
    def _longest_common_prefix_length(str1, str2):
        """Finds the length of the longest common prefix between two strings."""
        max_length = min(len(str1), len(str2))
        for i in range(max_length):
            if str1[i] != str2[i]:
                return i
        return max_length

if __name__ == "__main__":
    trie = RadixTree()

    # Add keys and values
    trie.add("apple", 1)
    trie.add("app", 2)
    trie.add("banana", 3)

    # Check if keys exist
    print(trie.contains("apple"))  # True
    print(trie.contains("app"))    # True
    print(trie.contains("banana")) # True
    print(trie.contains("orange")) # False

    # Retrieve values
    print(trie.get("apple"))  # 1
    print(trie.get("app"))    # 2
    print(trie.get("banana")) # 3
    print(trie.get("orange")) # None

    # Attempting to add an existing key
    try:
        trie.add("apple", 4)
    except ValueError as e:
        print(e)  # Key already exists in the trie.

    trie.save()

    trie = RadixTree()
    print(trie.contains("apple"))
    trie = RadixTree.from_disk()
    print(trie.contains("apple"))