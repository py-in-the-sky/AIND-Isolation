from functools import reduce


class BoardWrapper:
    """
    A hashable and __eq__ comparable representation of the board state.

    Exposes the Isolation board via the `board` attribute.
    """
    def __init__(self, board):
        self.board = board
        self._key = self._board_state()
        self._hash = hash(self._key)
        # This is a bit of a hack to keep _key and _hash as static values,
        # given board is mutable. However, boards aren't treated as mutable
        # values in the life of a call to alpha_beta, and therefore this
        # works.

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, BoardWrapper) and self._key == other._key

    def _board_state(self):
        return (self.board.width,
                self.board.height,
                self.board.get_player_location(self.board.active_player),
                self.board.get_player_location(self.board.inactive_player),
                self._board_bit_representation())

    def _board_bit_representation(self):
        "A compact bit_array representing all blank spaces of the board."

        def _mark_blank_space(bit_array, blank_space):
            "Mark blank space with a 1 in the bit_array."
            row, col = blank_space
            offset = row * self.board.width + col
            return bit_array | (1 << offset)

        return reduce(_mark_blank_space, self.board.get_blank_spaces(), 0)
