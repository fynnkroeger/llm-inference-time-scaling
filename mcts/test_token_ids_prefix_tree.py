import unittest
from math import log
from mcts.token_ids_prefix_tree import ExpectedValueSearchTree

class TestTokenIdsPrefixTree(unittest.TestCase):
    def test_probability_discounting_one_sequence(self):
        tree = ExpectedValueSearchTree()
        tree.add_sequence([0], [1, 2],[log(0.5), log(0.5)])
        sequence_root = tree.prompt_root[tuple([0])] 

        assert sequence_root["total_following_paths_probability"] == 0.25
        assert sequence_root.get_first_child()["total_following_paths_probability"] == 0.5

    def test_probability_discounting_two_sequences(self):
        tree = ExpectedValueSearchTree()
        tree.add_sequence([0], [1, 2],[log(0.5), log(0.5)])
        tree.add_sequence([0], [1, 3],[log(0.5), log(0.25)])
        sequence_root = tree.prompt_root[tuple([0])] 

        assert sequence_root["total_following_paths_probability"] == 0.5 * (0.5 + 0.25)
        assert sequence_root.get_first_child()["total_following_paths_probability"] == 0.5 + 0.25


    def test_probability_discounting_duplicate_sequence(self):
        tree = ExpectedValueSearchTree()
        tree.add_sequence([0], [1, 2],[log(0.5), log(0.5)])
        tree.add_sequence([0], [1, 2],[log(0.5), log(0.5)])

        sequence_root = tree.prompt_root[tuple([0])] 

        # Depending on the inference methods we might add the same seqeunce twice but it should only count once (same results as in test_probability_discounting_one_sequence)
        assert sequence_root["total_following_paths_probability"] == 0.25
        assert sequence_root.get_first_child()["total_following_paths_probability"] == 0.5


if __name__ == '__main__':
    unittest.main()