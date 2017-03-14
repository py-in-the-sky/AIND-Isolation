from collections import deque
from functools import reduce


BFS_DEPTH = 5
KNIGHT_DIRECTIONS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def moves(location, available):
    """Given a location on the board and all available (blank) spaces
    on the board, return all locations on the board that are valid moves.
    """
    if location is None:
        return available

    r, c = location
    moves = ((r+dr, c+dc) for dr,dc in KNIGHT_DIRECTIONS)
    valid_moves = (loc for loc in moves if loc in available)
    return valid_moves


### Heuristics.

def open_moves_depth_2(game, player):
    pass


def bfs_max_depth_heuristic(game, player):
    def _max_depth(p):
        location = game.get_player_location(p)
        visited = {}  # location: depth
        q = deque([ (location, 0) ])  # (location, depth)

        while q:
            loc, depth = q.popleft()
            if loc not in visited:
                visited[loc] = depth
                for loc2 in moves(loc, available):
                    if loc2 not in visited:
                        q.append((loc2, depth+1))

        return max(visited.values())

    available = set(game.get_blank_spaces())
    return float(_max_depth(player) - _max_depth(game.get_opponent(player)))


def bfs_open_moves_heuristic(game, player, bfs_depth=BFS_DEPTH):
    def _bfs_score(p):
        location = game.get_player_location(p)
        available = set(game.get_blank_spaces())
        visited = {}  # location: depth
        q = deque([ (location, 0) ])  # (location, depth)

        while q:
            loc, depth = q.popleft()

            if depth <= bfs_depth and loc not in visited:
                visited[loc] = depth

                for loc2 in moves(loc, available):
                    if loc2 not in visited:
                        q.append((loc2, depth+1))

        return float(sum(visited.values()))

    return _bfs_score(player) - _bfs_score(game.get_opponent(player))


def bfs_open_moves_with_blocking_heuristic(game, player, bfs_depth=BFS_DEPTH):
    opponent = game.get_opponent(player)
    player_location = game.get_player_location(player)
    opponent_location = game.get_player_location(player)
    available = set(game.get_blank_spaces())
    inf = float('inf')

    # BFS for player.
    player_score = 0
    player_visited = {}  # location: depth
    q = deque([ (player_location, 0) ])  # (location, depth)
    while q:
        loc, depth = q.popleft()
        if depth <= bfs_depth and loc not in player_visited:
            player_visited[loc] = depth
            player_score += depth
            for loc2 in moves(loc, available):
                if loc2 not in player_visited:
                    q.append((loc2, depth+1))

    # BFS for opponent.
    opponent_score = 0
    opponent_visited = {}  # location: depth
    q = deque([ (opponent_location, 0) ])
    while q:
        loc, depth = q.popleft()
        if depth <= bfs_depth and loc not in opponent_visited and depth < player_visited.get(loc, inf):
            opponent_visited[loc] = depth
            opponent_score += depth
            for loc2 in moves(loc, available):
                if loc2 not in opponent_visited:
                    q.append((loc2, depth+1))

    return float(player_score - opponent_score)


### Board Symmetries.

def reflect_vertical(location, board_width, _):
    "Reflect location across the vertical central axis of the board."
    r, c = location
    rightmost_column_of_board = board_width - 1
    return (r, rightmost_column_of_board - c)


def reflect_horizontal(location, _, board_height):
    "Reflect location across the horizontal central axis of the board."
    r, c = location
    bottom_row_of_board = board_height - 1
    return (bottom_row_of_board - r, c)


def reflect_secondary_diagonal(location, board_width, board_height):
    """Reflect location across the secondary diagonal of the board.
    The secondary diagonal spans from the upper-righthand corner to the
    bottom-lefthand corner of the board.
    """
    r, c = location
    rightmost_column_of_board = board_width - 1
    bottom_row_of_board = board_height - 1
    return (bottom_row_of_board - c, rightmost_column_of_board - r)


def reflect_primary_diagonal(location, board_width, board_height):
    """Reflect location across the primary diagonal of the board.
    The primary diagonal spans from the upper-lefthand corner to the
    bottom-righthand corner of the board.
    """
    l, bw, bh = location, board_width, board_height
    return reflect_secondary_diagonal(reflect_horizontal(reflect_vertical(l, bw, bh), bw, bh), bw, bh)


def rotate_90(location, board_width, board_height):
    l, bw, bh = location, board_width, board_height
    return reflect_secondary_diagonal(reflect_vertical(l, bw, bh), bw, bh)


def rotate_180(location, board_width, board_height):
    l, bw, bh = location, board_width, board_height
    return reflect_secondary_diagonal(reflect_primary_diagonal(l, bw, bh), bw, bh)


def rotate_270(location, board_width, board_height):
    l, bw, bh = location, board_width, board_height
    return rotate_90(rotate_180(l, bw, bh), bw, bh)


def test_reflections_and_rotations():
    "Property tests of the reflection and rotation functions."
    from random import randint, randrange

    for _ in range(250):
        w, h = randint(1, 15), randint(1, 15)  # Rectangular grid.
        r, c = randrange(0, h), randrange(0, w)
        loc = (r, c)
        assert reflect_vertical(reflect_vertical(loc, w, h), w, h) == loc
        assert reflect_horizontal(reflect_horizontal(loc, w, h), w, h) == loc
        assert rotate_180(rotate_180(loc, w, h), w, h) == loc

    for _ in range(250):
        h = w = randint(1, 15)  # Square grid.
        r, c = randrange(0, h), randrange(0, w)
        loc = (r, c)
        assert reflect_vertical(reflect_vertical(loc, w, h), w, h) == loc
        assert reflect_horizontal(reflect_horizontal(loc, w, h), w, h) == loc
        assert reflect_secondary_diagonal(reflect_secondary_diagonal(loc, w, h), w, h) == loc
        assert reflect_primary_diagonal(reflect_primary_diagonal(loc, w, h), w, h) == loc
        assert rotate_270(rotate_90(loc, w, h), w, h) == loc
        assert rotate_180(rotate_180(loc, w, h), w, h) == loc
        assert rotate_270(rotate_90(loc, w, h), w, h) == loc

    print('Tests pass!')


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


def board_symmetries(board):
    """
    REFLECT/ROTATE (RR) A BOARD

    * board.copy()
    * square vs rectangle cases
    * RR the moves in __last_player_move__ if not Board.NOT_MOVED
    * copy and RR/assign __board_state__
    """
    yield BoardWrapper(board)

    w, h = board.width, board.height

    symmetry_functions = [reflect_vertical, reflect_horizontal, rotate_180]
    is_square = (w == h)
    if is_square:
        symmetry_functions += [reflect_secondary_diagonal, reflect_primary_diagonal, rotate_90, rotate_270]

    for sf in symmetry_functions:
        new_board = board.copy()

        for player,move in board.__last_player_move__.items():
            if move is not board.NOT_MOVED:
                new_board.__last_player_move__[player] = sf(move, w, h)

        for row in range(h):
            for col in range(w):
                row2, col2 = sf((row, col), w, h)
                new_board.__board_state__[row2][col2] = board.__board_state__[row][col]

        yield BoardWrapper(new_board)


if __name__ == '__main__':
    test_reflections_and_rotations()
