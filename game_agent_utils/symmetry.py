from .BoardWrapper import BoardWrapper


def reflect_vertical(location, board_width, _):
    """Reflect location across the vertical central axis of the board.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_width: int
        The width of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to the reflection of `location`
        across the vertical central axis of the board.
    """
    r, c = location
    rightmost_column_of_board = board_width - 1
    return (r, rightmost_column_of_board - c)


def reflect_horizontal(location, _, board_height):
    """Reflect location across the horizontal central axis of the board.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_height: int
        The height of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to the reflection of `location`
        across the horizontal central axis of the board.
    """
    r, c = location
    bottom_row_of_board = board_height - 1
    return (bottom_row_of_board - r, c)


def reflect_secondary_diagonal(location, board_width, board_height):
    """Reflect location across the secondary diagonal of the board.
    The secondary diagonal spans from the upper-righthand corner to the
    bottom-lefthand corner of the board.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_width: int
        The width of the board.

    board_height: int
        The height of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to the reflection of `location`
        across the secondary diagonal of the board.
    """
    r, c = location
    rightmost_column_of_board = board_width - 1
    bottom_row_of_board = board_height - 1
    return (bottom_row_of_board - c, rightmost_column_of_board - r)


def reflect_primary_diagonal(location, board_width, board_height):
    """Reflect location across the primary diagonal of the board.
    The primary diagonal spans from the upper-lefthand corner to the
    bottom-righthand corner of the board.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_width: int
        The width of the board.

    board_height: int
        The height of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to the reflection of `location`
        across the primary diagonal of the board.
    """
    l, bw, bh = location, board_width, board_height
    return reflect_secondary_diagonal(reflect_horizontal(reflect_vertical(l, bw, bh), bw, bh), bw, bh)


def rotate_90(location, board_width, board_height):
    """Returns the coordinates of location after the board is
    rotated 90 degrees counter-clockwise.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_width: int
        The width of the board.

    board_height: int
        The height of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to `location` after the board
        is rotated 90 degrees counter-clockwise.
    """
    l, bw, bh = location, board_width, board_height
    return reflect_secondary_diagonal(reflect_vertical(l, bw, bh), bw, bh)


def rotate_180(location, board_width, board_height):
    """Returns the coordinates of location after the board is
    rotated 180 degrees counter-clockwise.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_width: int
        The width of the board.

    board_height: int
        The height of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to `location` after the board
        is rotated 180 degrees counter-clockwise.
    """
    l, bw, bh = location, board_width, board_height
    return reflect_secondary_diagonal(reflect_primary_diagonal(l, bw, bh), bw, bh)


def rotate_270(location, board_width, board_height):
    """Returns the coordinates of location after the board is
    rotated 270 degrees counter-clockwise.

    Parameters
    ----------
    location : (int, int)
        A (row, column) tuple, representing a location on the board.

    board_width: int
        The width of the board.

    board_height: int
        The height of the board.

    Returns
    -------
    (int, int)
        Board coordinates corresponding to `location` after the board
        is rotated 270 degrees counter-clockwise.
    """
    l, bw, bh = location, board_width, board_height
    return rotate_90(rotate_180(l, bw, bh), bw, bh)


def board_symmetries(board):
    """Given a game board, performs all possible reflections and rotations
    of the board, yielding each one, wrapped in a `BoardWrapper`.

    REFLECT/ROTATE (RR) a board:
    * board.copy()
    * square vs rectangle cases
    * RR the moves in __last_player_move__ if not Board.NOT_MOVED
    * copy and RR/assign __board_state__

    Parameters
    ----------
    board : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Yields
    -------
    BoardWrapper
        A hashable wrapper around the reflected/rotated board.
    """
    yield BoardWrapper(board)

    w, h = board.width, board.height

    symmetry_functions = [reflect_vertical, reflect_horizontal, rotate_180]
    board_is_square = (w == h)
    if board_is_square:
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
