"""
This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""


import game_agent_utils.heuristics as gau_heur
import game_agent_utils.symmetry as gau_symm

from collections import defaultdict
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # `player` is treated as the maximizing player for the game tree, not
    # necessarily as the player with the current move.
    score = gau_heur.who_can_get_there_first_depth_n(game, player, max_depth=5)
    return float(score)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.

    show_stats : boolean (optional)
        Whether to record statistics so that they can be show when the show_stats
        method is called.

    use_symmetries : boolean (optional)
        Whether to use board symmetries to prune the game tree at early plies in
        the game tree. Scores for games will be memoized and then reflections and
        rotations of boards will be searched to see whether an equivalent board
        state has been searched before.

    symmetry_threshold : boolean (optional)
        The depth (ply) of the game tree up to which symmetrically equivalent boards
        will be used to prune the game tree.

    use_rollouts : boolean (optional)
        Whether to use pure Monte Carlo rollouts low in the game tree (i.e., at
        late plies) to help estimate the value of an internal node of the game
        tree.

    try_reflection : boolean (optional)
        Whether to try to play the "reflection" strategy as an opening move.
        The strategy is simple: always play the 180-degree rotation of your
        opponent's last move.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, iterative=True, method='minimax',
                 timeout=10., show_stats=False, use_symmetries=False, symmetry_threshold=2,
                 use_rollouts=False, try_reflection=False):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = lambda game: score_fn(game, self)
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

        # Symmetries (i.e., reflections and rotations)
        self._symmetries_cache = None
        self._starting_ply = None
        self._n_squares = 0
        self.use_symmetries = use_symmetries
        self.symmetry_threshold = symmetry_threshold

        # Monte Carlo rollouts
        self.use_rollouts = use_rollouts

        # Opening moves
        self.try_reflection = try_reflection
        self._playing_reflection = False
        self._played_board_center = False

        # Stats
        # self._show_stats = show_stats
        # self.symmetry_cache_hits = defaultdict(list)  # ply: n_cache_hits
        # self.game_depths = []  # Ply at which winner is determined.
        # self.branching_factors = []  # Number of branches at each node.
        # self.branching_factors_by_ply = defaultdict(list)
        # self.ab_pruning_by_ply = defaultdict(list)
        # self.monte_carlo_scores_by_ply = defaultdict(lambda: defaultdict(int))
        # self.scores_by_ply = defaultdict(lambda: defaultdict(int))
        # self.search_depths = defaultdict(int)
        # self.rollout_depths = defaultdict(int)

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        self._n_squares = game.width * game.height
        new_starting_ply = self._n_squares - len(game.get_blank_spaces()) + 1
        self._starting_ply = new_starting_ply  # Starting ply starts from 1, not 0.
        new_game = (self._starting_ply is None) or (new_starting_ply < self._starting_ply + 2)
        # new_game: whether a new game has been passed to get_move or I'm continuing an ongoing game.
        # This is useful in the case of playing the "reflection" strategy: if I've committed to
        # playing that strategy, then I will continue doing so as long as a new game hasn't started.

        move = self._opening_book(game, new_game)
        if move is not None:
            return move

        best_move = (-1, -1)
        dynamic_search_depth = 0  # Used for iterative deepening.
        search_method = self.minimax if self.method == 'minimax' else self.alphabeta

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                while dynamic_search_depth <= self._n_squares:
                    # `dynamic_search_depth <= self._n_squares` is a quick check that the
                    # board could not have been filled out. This prevents the while loop
                    # from continuing to spin while not move can be made on the board,
                    # helping the matches to proceed more quickly.
                    self._symmetries_cache = {}
                    self.search_depth = dynamic_search_depth
                    _, best_move = search_method(game, dynamic_search_depth)
                    dynamic_search_depth += 1
            else:
                _, best_move = search_method(game, self.search_depth)
        except Timeout:
            pass

        # self.search_depths[dynamic_search_depth] += 1
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if game.is_winner(self):
            return float('inf'), (-1, -1)
        elif game.is_loser(self):
            return float('-inf'), (-1, -1)
        elif depth == 0:
            return self.score(game), (-1, -1)

        new_depth = depth - 1
        m_player = not maximizing_player
        optimizing_fn = max if maximizing_player else min
        score, move = optimizing_fn(((self.minimax(game.forecast_move(m), new_depth, m_player)[0], m)
                                     for m in game.get_legal_moves()),
                                    key=lambda pair: pair[0])
        self._cache_move(game, score, depth)
        return score, move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # The following are for gathering statistics:
        # ply = self._starting_ply + self.search_depth - depth
        # n_legal_moves = len(game.get_legal_moves())
        # self.branching_factors.append(n_legal_moves)
        # self.branching_factors_by_ply[ply].append(n_legal_moves)

        # Handle game ending or search depth maxed out:
        if game.is_winner(self):
            # self.game_depths.append(ply)
            return float('inf'), (-1, -1)
        elif game.is_loser(self):
            # self.game_depths.append(ply)
            return float('-inf'), (-1, -1)
        elif depth == 0:
            score = self._eval(game, depth)
            # _eval wraps self.score and optionally performs Monte Carlo rollouts.
            return score, (-1, -1)

        # See if equivalent board has already been scored:
        symmetry_score = self._check_symmetries(game, depth)
        if symmetry_score is not None and alpha < symmetry_score <= beta:
            return symmetry_score, (-1, -1)

        # Continue Alpha-Beta search:
        if maximizing_player:
            val, best_move, new_depth, m_player = float('-inf'), (-1, -1), depth-1, (not maximizing_player)

            for m in game.get_legal_moves():
                old_val = val
                val = max(val, self.alphabeta(game.forecast_move(m), new_depth, alpha, beta, m_player)[0])
                best_move = best_move if val == old_val else m
                # n_legal_moves -= 1

                if val >= beta:
                    # self.ab_pruning_by_ply[ply].append(n_legal_moves)
                    self._cache_move(game, val, depth)
                    return (val, best_move)

                alpha = max(alpha, val)

            # self.ab_pruning_by_ply[ply].append(n_legal_moves)
            self._cache_move(game, val, depth)
            return (val, best_move)
        else:
            val, best_move, new_depth, m_player = float('inf'), (-1, -1), depth-1, (not maximizing_player)

            for m in game.get_legal_moves():
                old_val = val
                val = min(val, self.alphabeta(game.forecast_move(m), new_depth, alpha, beta, m_player)[0])
                best_move = best_move if val == old_val else m
                # n_legal_moves -= 1

                if val <= alpha:
                    # self.ab_pruning_by_ply[ply].append(n_legal_moves)
                    self._cache_move(game, val, depth)
                    return (val, best_move)

                beta = min(beta, val)

            # self.ab_pruning_by_ply[ply].append(n_legal_moves)
            self._cache_move(game, val, depth)
            return (val, best_move)

    def _check_symmetries(self, game, depth):
        """See whether an equivalent game state has already been evaluated.
        Returns None if self.use_symmetries is False or the ply of the
        game state is > 2.

        Equivalent game states are all rotations and reflections of the
        game board and the players' current positions.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            The distance from this search's terminal search depth.

        Returns
        -------
        float or None
            Returns the score if an equivalent game state has been memoized
            and the self.use_symmetries flag is set to True. Otherwise, returns
            None.
        """
        if not self.use_symmetries:
            return None

        ply = self._starting_ply + self.search_depth - depth - 1

        if ply <= self.symmetry_threshold:
            board_wrapper = gau_symm.BoardWrapper(game)
            score = self._symmetries_cache.get(board_wrapper, None)
            # self.symmetry_cache_hits[ply].append(1 if score is not None else 0)
            return score

    def _cache_move(self, game, score, depth):
        """If the self.use_symmetries flag is set to True and the ply of the
        game state is <= 2, then this function will memoize the score for the
        game state and all equivalent game states.

        Equivalent game states are all rotations and reflections of the
        game board and the players' current positions.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        score : float
            The score for the game state.

        depth : int
            The distance from this search's terminal search depth.
        """
        if not self.use_symmetries:
            return None

        ply = self._starting_ply + self.search_depth - depth - 1

        if ply <= self.symmetry_threshold:
            for board_wrapper in gau_symm.board_symmetries(game):
                self._symmetries_cache[board_wrapper] = score

    def _eval(self, game, depth):
        """Returns a score for a node in the game tree.

        Use in place of self.score if you'd like to switch from heuristic
        scoring to Monte Carlo rollouts low in the game tree, where the
        branching factor is small.

        Will use rollouts if self.use_rollouts is True and the ply of the
        game state is >= 31.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            The distance from this search's terminal search depth.

        Returns
        -------
        float
            Returns an score for the game-tree node.
        """
        score = self.score(game)

        if not self.use_rollouts:
            return score

        ply = self._starting_ply + self.search_depth - depth

        if ply < 31:
            return score

        weight = 2.0  # 4.0
        n_rollouts = 14  # 7
        rollouts_score = sum(self._monte_carlo_rollout(game.copy()) for _ in range(n_rollouts))
        # self.monte_carlo_scores_by_ply[ply][rollouts_score] += 1
        # self.scores_by_ply[ply][score] += 1
        return score + weight * rollouts_score

    def _monte_carlo_rollout(self, game):
        """Conducts a pure Monte Carlo rollout of the game. Returns
        1 if self wins. Otherwise, returns -1.

        Will mutate game, so pass in a game.copy() if mutation is not desired.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        Returns
        -------
        int
            1 is self wins; otherwise, -1.
        """
        # depth = 0

        while True:
            if game.is_winner(self):
                # self.rollout_depths[depth] += 1
                return 1
            elif game.is_loser(self):
                # self.rollout_depths[depth] += 1
                return -1

            # depth += 1
            move = random.choice(game.get_legal_moves())  # Random legal move for active player.
            game.apply_move(move)  # Applies move to board and changes active player.

    def _opening_book(self, game, new_game):
        """Analyzes the game state and tries to return an intelligent
        opening move. If no move can be determined, returns None.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        new_game : boolean
            Indicates whether a new game is being played this round or an old
            one is being continued.

        Returns
        -------
        tuple(int, int) or None
            An opening move, if one can be determined. Otherwise, None.
        """
        if self.try_reflection:
            # My opening book will include trying to reflect the opponent's moves if my
            # try_reflection flag is True.
            move = self._reflection(game, new_game)

            if move is not None:
                return move

        if self._starting_ply == 1:
            return self._board_center(game)

        opp_loc = game.get_player_location(game.get_opponent(self))
        if self._starting_ply == 2 and opp_loc == self._board_center(game):
            # If ply = 2 and opponent occupies center, then make sure not to move into
            # one of the opponent's legal next moves. This prevents the opponent from
            # playing reflection.
            my_moves = set(game.get_legal_moves(self))
            opp_moves = set(game.get_legal_moves(game.get_opponent(self)))
            choices = my_moves - opp_moves

            if choices:
                return random.choice(list(choices))

    def _reflection(self, game, new_game):
        """Returns reflection (i.e., 180-degree rotation) of opponent's last move if
        playing reflection is a valid strategy. Otherwise, returns None.

        As an opening move, we play reflection if possible because it guarantees a win.
        That is, if we can alway mirror the opponent's last move, then the opponent must
        be the first player to become isolated.

        The only time this can fail is if the opponent has the first move and doesn't move
        into center. Then it's conceivable that one of the opponent's future moves will be
        to the center of the board, which is a move that cannot be reflected.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        new_game : boolean
            Indicates whether a new game is being played this round or an old
            one is being continued.

        Returns
        -------
        tuple(int, int) or None
            A move using reflection, if possible. Otherwise, None.
        """
        if new_game:
            # With a new game, we don't know yet whether we can play reflection, so set to False.
            self._playing_reflection = False
            self._played_board_center = False

        if self._playing_reflection:
            return self._reflect(game)
        elif self._starting_ply == 1 and self._has_center(game):
            # I'm first mover of new game, and board has a center (i.e., board has odd
            # height and odd width).
            self._played_board_center = True
            return self._board_center(game)
        elif self._starting_ply == 2 and self._can_reflect(game):
            # If I'm second mover of game and can reflect opponent's current move, play reflection.
            self._playing_reflection = True
            return self._reflect(game)
        elif self._starting_ply == 3 and self._played_board_center and self._can_reflect(game):
            # If I'm first mover of game and I know I've played center of board, then if I can
            # reflect opponent's current move, I play reflection.
            self._playing_reflection = True
            return self._reflect(game)

    def _reflect(self, game):
        """180-degree rotation of opponent's last move.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        Returns
        -------
        tuple(int, int)
            A (row, column) tuple that is a 180-degree rotation of the opponent's
            last move.
        """
        opp_move = game.get_player_location(game.get_opponent(self))
        return gau_symm.rotate_180(opp_move, game.width, game.height)

    def _has_center(self, game):
        "True if board has odd height and width. Otherwise, False."
        return (game.width % 2 == 1) and (game.height % 2 == 1)

    def _board_center(self, game):
        """Return center square of the board. If board does not have odd height and width,
        then this function will return the bottom-right of the center cluster of squares."""
        row = game.height // 2
        col = game.width // 2
        return (row, col)

    def _can_reflect(self, game):
        """True if the 180-degree rotation of the opponent's last move is a
        legal move for me. Otherwise, False.
        """
        return self._reflect(game) in game.get_legal_moves(self)

    # def show_stats(self):
    #     if not self._show_stats:
    #         return None

    #     print('Rollout depths reached:')
    #     for depth,count in sorted(self.rollout_depths.items()):
    #         print('    Depth:', depth, 'Count:', count)

    #     print('Search depths reached:')
    #     for depth,count in sorted(self.search_depths.items()):
    #         print('    Depth:', depth, 'Count:', count)

    #     print()
    #     print('Game depth:',
    #           'Avg:', sum(self.game_depths) / len(self.game_depths),
    #           'Min:', min(self.game_depths),
    #           'Max:', max(self.game_depths))

    #     print()
    #     print('Branching factor',
    #           'Avg:', sum(self.branching_factors) / len(self.branching_factors),
    #           'Min:', min(self.branching_factors),
    #           'Max:', max(self.branching_factors))

    #     print()
    #     print('Branching factors by ply:')
    #     for ply,branching_factors in sorted(self.branching_factors_by_ply.items()):
    #         print('    Ply:', ply,
    #               'Avg:', sum(branching_factors) / len(branching_factors),
    #               'Min:', min(branching_factors),
    #               'Max:', max(branching_factors))

    #     print()
    #     print('AB pruning by ply:')
    #     for ply,pruning_counts in sorted(self.ab_pruning_by_ply.items()):
    #         print('    Ply:', ply,
    #               'Avg:', sum(pruning_counts) / len(pruning_counts),
    #               'Min:', min(pruning_counts),
    #               'Max:', max(pruning_counts))

    #     print()
    #     print('Symmetry cache hits by ply:')
    #     for ply,hits in sorted(self.symmetry_cache_hits.items()):
    #         n_hits = sum(hits)

    #         if n_hits > 0:
    #             print('    Ply:', ply,
    #                   'Hit ratio:', n_hits / len(hits),
    #                   'Total hits:', n_hits)

    #     print()
    #     print('Monte Carlo rollout distributions by ply:')
    #     for ply,dist in sorted(self.monte_carlo_scores_by_ply.items()):
    #         print('    Ply:', ply, 'Distribution:', sorted(dist.items()))

    #     print()
    #     print('Heuristic score distributions by ply:')
    #     for ply,dist in sorted(self.scores_by_ply.items()):
    #         print('    Ply:', ply, 'Distribution:', sorted(dist.items()))
