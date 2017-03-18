from collections import deque


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


def open_moves_depth_n(game, player, max_depth=2):
    def _bfs_score(p):
        location = game.get_player_location(p)
        visited = {}  # location: depth
        q = deque([ (location, 0) ])  # (location, depth)

        while q:
            loc, depth = q.popleft()
            if depth <= max_depth and loc not in visited:
                visited[loc] = depth
                for loc2 in moves(loc, available):
                    if loc2 not in visited:
                        q.append((loc2, depth+1))

        return sum(visited.values())

    available = set(game.get_blank_spaces())
    return float(_bfs_score(player) - _bfs_score(game.get_opponent(player)))


def interleaved_bfs_depth_n(game, player, max_depth=4):
    def _bfs(pA, pI):
        score = 0
        locA, locI = game.get_player_location(pA), game.get_player_location(pI)
        q = deque([(locA, 1, 0), (locI, -1, 0)])
        visited = set()

        while q:
            loc, weight, depth = q.popleft()
            if depth <= max_depth and loc not in visited:
                visited.add(loc)
                score += weight * depth
                for loc2 in moves(loc, available):
                    if loc2 not in visited:
                        q.append((loc2, weight, depth+1))

        return score

    weight = 1 if player == game.active_player else -1
    available = set(game.get_blank_spaces())
    return weight * _bfs(game.active_player, game.inactive_player)


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


def bfs_open_moves_heuristic(game, player, bfs_depth=5):
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

        return sum(visited.values())

    return float(_bfs_score(player) - _bfs_score(game.get_opponent(player)))


def bfs_open_moves_with_blocking_heuristic(game, player, bfs_depth=5):
    active_player, inactive_player = game.active_player, game.inactive_player
    available = set(game.get_blank_spaces())
    inf = float('inf')

    # BFS for active_player (first mover).
    active_player_score = 0
    active_player_visited = {}  # location: depth
    q = deque([ (game.get_player_location(active_player), 0) ])  # (location, depth)
    while q:
        loc, depth = q.popleft()
        if depth <= bfs_depth and loc not in active_player_visited:
            active_player_visited[loc] = depth
            active_player_score += depth
            for loc2 in moves(loc, available):
                if loc2 not in active_player_visited:
                    q.append((loc2, depth+1))

    # BFS for inactive_player (second mover).
    inactive_player_score = 0
    inactive_player_visited = {}  # location: depth
    q = deque([ (game.get_player_location(inactive_player), 0) ])
    while q:
        loc, depth = q.popleft()
        if depth <= bfs_depth and loc not in inactive_player_visited and depth < active_player_visited.get(loc, inf):
            inactive_player_visited[loc] = depth
            inactive_player_score += depth
            for loc2 in moves(loc, available):
                if loc2 not in inactive_player_visited:
                    q.append((loc2, depth+1))

    if player is active_player:
        return float(active_player_score - inactive_player_score)
    else:
        return float(inactive_player_score - active_player_score)
