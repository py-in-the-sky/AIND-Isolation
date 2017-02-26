from collections import deque


### Heuristics

def improved_score_depth_n(game, player, max_depth=5):
    """Like the improved-score heuristic, except that it considers not just
    the immediate squares available to a player but also the squares that are
    one, two, ..., max_depth moves away from the player.

    Each square that figures into a player's score is weighted by the number of
    moves (i.e., search depth) it takes for the player to get there. Therefore,
    this scoring favors players that can potentially make many subsequent moves
    in the future. The heuristic uses breadth-first search to determine how
    many moves there are between a square and a player's current position.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    max_depth : integer
        Controls how deep the search algorithm will go to find squares that are
        open to a player. E.g., if max_depth = 2, then the algorithm will count
        all squares that are one and two hops away from a player's current
        position.

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    def _bfs_score(p):
        location = game.get_player_location(p)
        visited = {}  # location: depth
        q = deque([ (location, 0) ])  # (location, depth)

        while q:
            loc, depth = q.popleft()
            if depth <= max_depth and loc not in visited:
                visited[loc] = depth
                for loc2 in _moves(loc, available):
                    if loc2 not in visited:
                        q.append((loc2, depth+1))

        return sum(visited.values())

    available = set(game.get_blank_spaces())
    return float(_bfs_score(player) - _bfs_score(game.get_opponent(player)))


def who_can_get_there_first(game, player):
    """Similar to the improved_score function. However, the active player's
    (the player with the next move) open moves cannot count towards the
    inactive player's open moves. It's as if the active player can take those
    squares first and block the inactive player from moving to them.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    active_moves = game.get_legal_moves(game.active_player)
    inactive_moves = [m for m in game.get_legal_moves(game.inactive_player) if m not in active_moves]

    own_moves = active_moves if player is game.active_player else inactive_moves
    opp_moves = inactive_moves if own_moves is active_moves else active_moves

    return float(len(own_moves) - len(opp_moves))


def who_can_get_there_first_depth_n(game, player, max_depth=5):
    """Similar in spirit to improved_score_depth_n and who_can_get_there_first.
    If a player can get to a square first in the breadth-first search, then
    that square cannot count as an open square for the opponent.

    This is how we determine who can get to a square first: a breadth-first
    search is conducted, where the active player (the one who has the next
    move in the game) first gets to move to all squares that are one move away.
    Then those squares are not available to the inactive player, who then gets to
    move to all squares that are one move away from his current position that have
    not been consumed by the active player. Then the active player gets to move to
    all squares that are two moves away, making those unavailable to the inactive
    player. And so on.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    max_depth : integer
        Controls how deep the search algorithm will go to find squares that are
        open to a player. E.g., if max_depth = 2, then the algorithm will count
        all squares that are one and two hops away from a player's current
        position.

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    weight = 1 if player == game.active_player else -1
    return weight * _interleaved_bfs_depth_n(game, max_depth)


def bfs_max_depth_heuristic(game, player):
    """Perfoms a breadth-first search for each player from the player's
    current position over all blank squares on the board. Outputs the
    difference between the maximum search depth that `player` and `player`'s
    opponent can achieve.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    def _max_depth(p):
        location = game.get_player_location(p)
        visited = {}  # location: depth
        q = deque([ (location, 0) ])  # (location, depth)

        while q:
            loc, depth = q.popleft()
            if loc not in visited:
                visited[loc] = depth
                for loc2 in _moves(loc, available):
                    if loc2 not in visited:
                        q.append((loc2, depth+1))

        return max(visited.values())

    available = set(game.get_blank_spaces())
    return float(_max_depth(player) - _max_depth(game.get_opponent(player)))


def dfs_max_depth_heuristic(game, player):
    """Like bfs_max_depth_heuristic but uses depth-first search instead of
    breadth-first, measuring the longest unobstructed path that each player
    has. Returns the difference of these lengths.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    def _max_depth(p, move=None):
        if move not in available:
            return 0

        move = move or game.get_player_location(p)
        available.discard(move)
        return 1 + max(_max_depth(p, m) for m in _moves(move, available))

    available = set(game.get_blank_spaces())
    own_max_depth = _max_depth(player)

    available = set(game.get_blank_spaces())
    opp_max_depth = _max_depth(game.get_opponent(player))

    return float(own_max_depth - opp_max_depth)


### Utilities

KNIGHT_DIRECTIONS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def _moves(location, available):
    """Given a board location and all available (blank) spaces on the board,
    return all board locations that are valid moves.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple representing a location on the board.

    available : set
        A set of all (row, column) tuples that are blank on the board.

    Returns
    ----------
    [(int, int)]
        A list of (row, column) tuples that are valid knight moves from location.
    """
    if location is None:
        return available

    r, c = location
    moves = ((r+dr, c+dc) for dr,dc in KNIGHT_DIRECTIONS)
    valid_moves = (loc for loc in moves if loc in available)
    return valid_moves


def _interleaved_bfs_depth_n(game, max_depth=4):
    """Scores the game board from the perspective of the active player.

    Every square that the active player can reach first in the breadth-first
    search contributes `depth` to the score, where `depth` is the search
    depth of that square from the active player's current position.

    Every square that the inactive player can reach first in the breadth-first
    search contributes `-1 * depth` to the score.

    We use `depth` to contribute to the score, so that squares that are farther
    away from a player's current position on the board contribute more to the
    score. This scoring mechanism favors players that have many moves left in
    their future.

    The active player, who is the next player to move in the game, gets first-mover
    advantage in the breadh-first search.

    A positive score means that the active player has more squares closer to him
    and has a better chance of cutting off the movement of the inactive player.
    Similarly, a negative score means that the inactive player has more squares
    closer to him and has a better chance of cutting of the movement of the
    active player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    max_depth : integer
        Controls how deep the search algorithm will go to find squares that are
        open to a player. E.g., if max_depth = 2, then the algorithm will count
        all squares that are one and two hops away from a player's current
        position.

    Returns
    ----------
    int
        A score for the current game state
    """
    score = 0
    locA = game.get_player_location(game.active_player)
    locI = game.get_player_location(game.inactive_player)
    q = deque([ (locA, 1, 0), (locI, -1, 0) ])  # Tuples of (location, weight, depth).
    available = set(game.get_blank_spaces())

    while q:
        loc, weight, depth = q.popleft()
        if loc in available or loc in (locA, locI):
            available.discard(loc)
            score += weight * depth
            if depth < max_depth:
                for loc2 in _moves(loc, available):
                    if loc2 in available:
                        q.append((loc2, weight, depth+1))

    return score



# def bfs_open_moves_with_blocking_heuristic(game, player, bfs_depth=5):
#     """Similar to who_can_get_there_first but more complicated and less
#     effective.

#     It only allows the active player to block the inactive player, whereas
#     who_can_get_there_first allows each player to block the other.

#     It was the first iteration on what eventually led to who_can_get_there_first.
#     """
#     active_player, inactive_player = game.active_player, game.inactive_player
#     available = set(game.get_blank_spaces())
#     inf = float('inf')

#     # BFS for active_player (first mover).
#     active_player_score = 0
#     active_player_visited = {}  # location: depth
#     q = deque([ (game.get_player_location(active_player), 0) ])  # (location, depth)
#     while q:
#         loc, depth = q.popleft()
#         if depth <= bfs_depth and loc not in active_player_visited:
#             active_player_visited[loc] = depth
#             active_player_score += depth
#             for loc2 in _moves(loc, available):
#                 if loc2 not in active_player_visited:
#                     q.append((loc2, depth+1))

#     # BFS for inactive_player (second mover).
#     inactive_player_score = 0
#     inactive_player_visited = {}  # location: depth
#     q = deque([ (game.get_player_location(inactive_player), 0) ])
#     while q:
#         loc, depth = q.popleft()
#         if depth <= bfs_depth and loc not in inactive_player_visited and depth < active_player_visited.get(loc, inf):
#             inactive_player_visited[loc] = depth
#             inactive_player_score += depth
#             for loc2 in _moves(loc, available):
#                 if loc2 not in inactive_player_visited:
#                     q.append((loc2, depth+1))

#     if player is active_player:
#         return float(active_player_score - inactive_player_score)
#     else:
#         return float(inactive_player_score - active_player_score)
