"""
TODO: Optimizations
    * Symmetries
    * Iterative deepening: order exploration of children based on last iteration's
      scores for the children (suggestion from AIMA)
    * Represent board as bit array (i.e., integer) where square (0, 0) is the
      least significant bit, and a 1 represents an unavailable square.
    * Iterative deepening: keep going until 50ms left.

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
    # NB: `player` is treated as the maximizing player for the game tree, not as
    # the player with the current move.
    # score = gau_heur.open_moves_depth_n(game, player, max_depth=4)
    score = gau_heur.interleaved_bfs_depth_n(game, player, max_depth=4)  # 5 seems good, too.
    # score = gau_heur.bfs_open_moves_with_blocking_heuristic(game, player)
    # score = gau_heur.bfs_open_moves_heuristic(game, player)
    # score = gau_heur.bfs_max_depth_heuristic(game, player)
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
    """

    def __init__(self, search_depth=3, score_fn=custom_score, iterative=True, method='minimax',
                 timeout=10., show_stats=False, use_symmetries=False, use_rollouts=False,
                 try_reflection=False):
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

        # Monte Carlo rollouts
        self.use_rollouts = use_rollouts

        # Opening moves
        self.try_reflection = try_reflection
        self._playing_reflection = False

        # Stats
        # self._show_stats = show_stats
        # self.symmetry_cache_hits = defaultdict(list)  # ply: n_cache_hits
        # self.game_depths = []  # Ply at which winner is determined.
        # self.branching_factors = []  # Number of branches at each node.
        # self.branching_factors_by_ply = defaultdict(list)
        # self.ab_pruning_by_ply = defaultdict(list)
        # self.monte_carlo_scores_by_ply = defaultdict(lambda: defaultdict(int))
        # self.scores_by_ply = defaultdict(lambda: defaultdict(int))

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

        if self.try_reflection:
            # My opening book will include trying to reflect the opponent's moves if my
            # try_reflection flag is True.
            new_game = (self._starting_ply is None) or (new_starting_ply < self._starting_ply + 2)

            if new_game:
                self._playing_reflection = False  # We don't know yet whether we can play reflection, so set to False.

            if self._playing_reflection:
                return self._reflect(game)
            elif self._starting_ply == 1 and self._has_center(game):  # I'm first mover of new game.
                return self._board_center(game)
            elif self._starting_ply == 2 and self._can_reflect(game):
                # If I'm second mover of game and can reflect opponent's current move, play reflection.
                self._playing_reflection = True
                return self._reflect(game)
            elif self._starting_ply == 3 and self._has_center(game) and self._can_reflect(game):
                # If I'm first mover of game, I know I've played center of board, so if I can
                # reflect opponent's current move, I initialize a game of reflection.
                self._playing_reflection = True
                return self._reflect(game)

        best_move = (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == 'minimax':
                if self.iterative:
                    d = 0
                    while d <= self._n_squares:
                        self._symmetries_cache = {}
                        self.search_depth = d
                        _, best_move = self.minimax(game, d)
                        d += 1
                else:
                    _, best_move = self.minimax(game, self.search_depth)
            else:
                if self.iterative:
                    d = 0
                    while d <= self._n_squares:
                        self._symmetries_cache = {}
                        self.search_depth = d
                        _, best_move = self.alphabeta(game, d)
                        d += 1
                else:
                    _, best_move = self.alphabeta(game, self.search_depth)
        except Timeout:
            pass

        # Return the best move from the last completed search iteration.
        return best_move

    def _reflect(self, game):
        pass

    def _board_center(self, game):
        pass

    def _can_reflect(self, game):
        pass

    def _has_center(self, game):
        pass

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
            score = self.score(game)
            return score, (-1, -1)

        opt_fn = max if maximizing_player else min
        d, m_player = depth-1, not maximizing_player
        score, move = opt_fn(((self.minimax(game.forecast_move(m), d, m_player)[0], m)
                               for m in game.get_legal_moves()), key=lambda pair: pair[0])
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

        # ply = self._starting_ply + self.search_depth - depth
        # n_legal_moves = len(game.get_legal_moves())
        # self.branching_factors.append(n_legal_moves)
        # self.branching_factors_by_ply[ply].append(n_legal_moves)

        if game.is_winner(self):
            # self.game_depths.append(ply)
            # if self.iterative and (self._starting_ply <= 3 or ply - self._starting_ply >= 15):
            #     print('Leaf node. Starting ply:', self._starting_ply, 'Current ply:', ply)
            return float('inf'), (-1, -1)
        elif game.is_loser(self):
            # self.game_depths.append(ply)
            # if self.iterative and (self._starting_ply <= 3 or ply - self._starting_ply >= 15):
            #     print('Leaf node. Starting ply:', self._starting_ply, 'Current ply:', ply)
            return float('-inf'), (-1, -1)
        elif depth == 0:
            score = self._eval(game, depth)
            return score, (-1, -1)

        symmetry_score = self._check_symmetries(game, depth)

        if symmetry_score is not None and alpha < symmetry_score <= beta:
            return symmetry_score, (-1, -1)
        elif maximizing_player:
            val = float('-inf')
            best_move = (-1, -1)

            for m in game.get_legal_moves():
                v = max(val, self.alphabeta(game.forecast_move(m), depth-1, alpha, beta, not maximizing_player)[0])
                best_move = best_move if v == val else m
                val = v
                # n_legal_moves -= 1

                if val >= beta:
                    # self.ab_pruning_by_ply[ply].append(n_legal_moves)
                    self._cache_move(game, val, depth)
                    return (val, best_move)

                alpha = max(alpha, val)

            # assert n_legal_moves == 0
            # self.ab_pruning_by_ply[ply].append(n_legal_moves)
            self._cache_move(game, val, depth)
            return (val, best_move)
        else:
            val = float('inf')
            best_move = (-1, -1)

            for m in game.get_legal_moves():
                v = min(val, self.alphabeta(game.forecast_move(m), depth-1, alpha, beta, not maximizing_player)[0])
                best_move = best_move if v == val else m
                val = v
                # n_legal_moves -= 1

                if val <= alpha:
                    # self.ab_pruning_by_ply[ply].append(n_legal_moves)
                    self._cache_move(game, val, depth)
                    return (val, best_move)

                beta = min(beta, val)

            # assert n_legal_moves == 0
            # self.ab_pruning_by_ply[ply].append(n_legal_moves)
            self._cache_move(game, val, depth)
            return (val, best_move)

    def _check_symmetries(self, game, depth):
        if not self.use_symmetries:
            return None

        ply = self._starting_ply + self.search_depth - depth - 1

        if ply <= 2:
            board_wrapper = gau_symm.BoardWrapper(game)
            score = self._symmetries_cache.get(board_wrapper, None)
            # self.symmetry_cache_hits[ply].append(1 if score is not None else 0)
            return score

    def _cache_move(self, game, score, depth):
        if not self.use_symmetries:
            return None

        ply = self._starting_ply + self.search_depth - depth - 1

        if ply <= 2:
            for board_wrapper in gau_symm.board_symmetries(game):
                self._symmetries_cache[board_wrapper] = score

    def _eval(self, game, depth):
        """Use in place of self.score if you'd like to switch from heuristic
        scoring to Monte Carlo rollouts low in the game tree, where the branching
        factor is small.
        """
        if not self.use_rollouts:
            return self.score(game)

        ply = self._starting_ply + self.search_depth - depth

        if ply >= 31:
            weight = 2.0  # 4.0
            n_rollouts = 14  # 7
            rollouts_score = sum(self._monte_carlo_rollout(game.copy()) for _ in range(n_rollouts))
            score = self.score(game)
            # self.monte_carlo_scores_by_ply[ply][rollouts_score] += 1
            # self.scores_by_ply[ply][score] += 1
            return score + weight * rollouts_score

    def _monte_carlo_rollout(self, game):
        """Conducts a pure Monte Carlo rollout of the game. Returns
        1.0 if self wins. Otherwise, returns -1.0.
        """
        while True:
            if game.is_winner(self):
                return 1
            elif game.is_loser(self):
                return -1

            move = random.choice(game.get_legal_moves())  # Random legal move for active player.
            game.apply_move(move)  # Applies move to board and changes active player.

    # def show_stats(self):
    #     if not self._show_stats:
    #         return None

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
