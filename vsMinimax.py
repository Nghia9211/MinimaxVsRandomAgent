import time
import random
import copy
from gym_xiangqi.agents import RandomAgent
from gym_xiangqi.constants import (     # NOQA
    RED, BLACK, PIECE_ID_TO_NAME, ALLY
)
from gym_xiangqi.utils import action_space_to_move,move_to_action_space
from gym_xiangqi.envs import XiangQiEnv

import threading
ROWS, COLS = 10, 9

piece_map = {
    9: "bR", 7: "bH", 5: "bE", 3: "bA", 1: "bK", 2: "bA", 4: "bE", 6: "bH", 8: "bR",
    10: "bC", 11: "bC",
    12: "bS", 13: "bS", 14: "bS", 15: "bS", 16: "bS",
    0: "--",
    -12: "rS", -13: "rS", -14: "rS", -15: "rS", -16: "rS",
    -10: "rC", -11: "rC",
    -8: "rR", -6: "rH", -4: "rE", -2: "rA", -1: "rK", -3: "rA", -5: "rE", -7: "rH", -9: "rR"
}


def flip_move(move):
    """ Đảo ngược nước đi để phù hợp với phía người chơi """
    start, end = move
    flipped_start = (9 - start[0], 8 - start[1])
    flipped_end = (9 - end[0], 8 - end[1])
    return flipped_start, flipped_end

def get_piece_at_position(board, position):
    """ Trả về quân cờ tại vị trí được chỉ định """
    row, col = position
    return board[row][col]



def flip_chessboard_properly(board):
    return [row[::-1] for row in board[::-1]]  # Lật bàn cờ cả theo chiều dọc và ngang



def convert_chessboard(numeric_board):
    return [[piece_map[num] for num in row] for row in numeric_board]

# Piece definitions (initial positions)
INITIAL_BOARD = [
    ["bR", "bH", "bE", "bA", "bK", "bA", "bE", "bH", "bR"],
    ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
    ["--", "bC", "--", "--", "--", "--", "--", "bC", "--"],
    ["bS", "--", "bS", "--", "bS", "--", "bS", "--", "bS"],
    ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
    ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
    ["rS", "--", "rS", "--", "rS", "--", "rS", "--", "rS"],
    ["--", "rC", "--", "--", "--", "--", "--", "rC", "--"],
    ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
    ["rR", "rH", "rE", "rA", "rK", "rA", "rE", "rH", "rR"]
]


from zobrist import ZOBRIST_TABLE
HASH_CACHE = {}
PIECE_INDEX = {'rK': 0, 'rA': 1, 'rE': 2, 'rH': 3,  'rR': 4,  'rC': 5,  'rS': 6,
               'bK': 7, 'bA': 8, 'bE': 9, 'bH': 10, 'bR': 11, 'bC': 12, 'bS': 13}

# Opening book for the first two moves
OPENING_BOOK = {
    # (7, 1) -> (7, 4)
    12174446791962688221:   [((0, 7), (2, 6)), ((0, 1), (2, 2))],
    # (7, 1) -> (7, 3)
    4624035311603896537:    [((2, 7), (2, 4)), ((3, 2), (4, 2))],
    # (6, 2) -> (5, 2)
    12150175912188942269:   [((2, 1), (2, 2)), ((3, 6), (4, 6))],

    # (7, 7) -> (7, 4)
    13107710493949774785:   [((0, 7), (2, 6)), ((0, 1), (2, 2))],
    # (7, 7) -> (7, 5)
    14188411443009226568:   [((2, 1), (2, 4)), ((3, 6), (4, 6))],
    # (6, 6) -> (5, 6)
    9314984010439949747:    [((2, 7), (2, 6)), ((3, 2), (4, 2))],
}

def compute_zobrist_hash(board):
    h = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece != "--":
                index = PIECE_INDEX[piece]
                h ^= ZOBRIST_TABLE[r][c][index]
    return h

def get_opening_move(board):
    key = compute_zobrist_hash(board)
    if key in OPENING_BOOK:
        return random.choice(OPENING_BOOK[key])
    return None

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def in_palace(r, c, color):
    return (3 <= c <= 5 and ((7 <= r <= 9) if color == 'r' else (0 <= r <= 2)))

def is_empty_or_enemy(board, r, c, color):
    return board[r][c] == "--" or board[r][c][0] != color

def get_all_moves(board, turn):
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece == "--" or piece[0] != turn:
                continue
            ptype = piece[1]

            if ptype == 'K':
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc) and in_palace(nr, nc, turn) and is_empty_or_enemy(board, nr, nc, turn):
                        moves.append(((r, c), (nr, nc)))

            elif ptype == 'A':
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc) and in_palace(nr, nc, turn) and is_empty_or_enemy(board, nr, nc, turn):
                        moves.append(((r, c), (nr, nc)))

            elif ptype == 'E':
                for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                    nr, nc = r + dr, c + dc
                    mr, mc = r + dr // 2, c + dc // 2
                    if in_bounds(nr, nc) and is_empty_or_enemy(board, nr, nc, turn) and board[mr][mc] == "--":
                        if (turn == 'r' and nr >= 5) or (turn == 'b' and nr <= 4):
                            moves.append(((r, c), (nr, nc)))

            elif ptype == 'H':
                for (dr, dc), (br, bc) in [((-2, -1), (-1, 0)), ((-2, 1), (-1, 0)),
                                           ((2, -1), (1, 0)), ((2, 1), (1, 0)),
                                           ((-1, -2), (0, -1)), ((-1, 2), (0, 1)),
                                           ((1, -2), (0, -1)), ((1, 2), (0, 1))]:
                    nr, nc = r + dr, c + dc
                    br, bc = r + br, c + bc
                    if in_bounds(nr, nc) and board[br][bc] == "--" and is_empty_or_enemy(board, nr, nc, turn):
                        moves.append(((r, c), (nr, nc)))

            elif ptype == 'C':
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    jumped = False
                    for i in range(1, 10):
                        nr, nc = r + dr * i, c + dc * i
                        if not in_bounds(nr, nc):
                            break
                        if board[nr][nc] == "--":
                            if not jumped:
                                moves.append(((r, c), (nr, nc)))
                        else:
                            if not jumped:
                                jumped = True
                            else:
                                if board[nr][nc][0] != turn:
                                    moves.append(((r, c), (nr, nc)))
                                break

            elif ptype == 'R':
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for i in range(1, 10):
                        nr, nc = r + dr * i, c + dc * i
                        if not in_bounds(nr, nc):
                            break
                        if board[nr][nc] == "--":
                            moves.append(((r, c), (nr, nc)))
                        else:
                            if board[nr][nc][0] != turn:
                                moves.append(((r, c), (nr, nc)))
                            break

            elif ptype == 'S':
                dr = -1 if turn == 'r' else 1
                river_row = 4 if turn == 'r' else 5

                # Forward move
                nr = r + dr
                nc = c
                if in_bounds(nr, nc) and is_empty_or_enemy(board, nr, nc, turn):
                    moves.append(((r, c), (nr, nc)))

                # Sideways moves after crossing the river
                if (turn == 'r' and r <= river_row) or (not turn == 'r' and r >= river_row):
                    for dc in [-1, 1]:
                        nr = r
                        nc = c + dc
                        if in_bounds(nr, nc) and is_empty_or_enemy(board, nr, nc, turn):
                            moves.append(((r, c), (nr, nc)))

    return moves

def make_move(board, move):
    new_board = copy.deepcopy(board)
    (sr, sc), (er, ec) = move
    new_board[er][ec] = new_board[sr][sc]
    new_board[sr][sc] = "--"
    return new_board

def flying_general_illegal(board):
    rK = None
    bK = None
    for r in range(10):
        for c in range(9):
            piece = board[r][c]
            if piece == 'rK':
                rK = (r, c)
            elif piece == 'bK':
                bK = (r, c)

    if rK and bK and rK[1] == bK[1]:  # same file
        col = rK[1]
        top = min(rK[0], bK[0]) + 1
        bottom = max(rK[0], bK[0])
        for row in range(top, bottom):
            if board[row][col] != '--':
                return False
        return True  # No pieces between = illegal

    return False

def is_in_check(board, turn):
    general_pos = None
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece == f'{turn}K':
                general_pos = (r, c)
                break
        if general_pos:
            break

    enemy_color = opposite_color(turn)
    enemy_moves = get_all_moves(board, enemy_color)
    for move in enemy_moves:
        if move[1] == general_pos:
            return True
    return False

def is_checkmate(board, turn):
    if not is_in_check(board, turn):
        return False

    moves = get_all_moves(board, turn)
    for move in moves:
        new_board = make_move(board, move)
        if not is_in_check(new_board, turn) and not flying_general_illegal(new_board):
            return False

    return True

def get_legal_moves(board, turn):
    moves = get_all_moves(board, turn)
    return moves
    # legal_moves = []
    # for move in moves:
    #     new_board = make_move(board, move)
    #     if not is_in_check(new_board, turn) and not flying_general_illegal(new_board):
    #         legal_moves.append(move)
    # return legal_moves

PIECE_VALUES = {
    'K': 10000, 'A': 110, 'E': 110, 'R': 600,
    'H': 270, 'C': 300, 'S': 70
}

ADVANCEMENT_BONUS = {
    'S': 1,
    'C': 0.5,
    'R': 0.5
}

CENTER_CONTROL = {(4, 4), (4, 3), (4, 5), (5, 4), (5, 3), (5, 5)}

POSITION_TABLES = {
    'K': [  # King
        [0, 0,  0, -5, -10, -5,  0, 0, 0],
        [0, 0,  0, -2,  -5, -2,  0, 0, 0],
        [0, 0,  0,  0,   0,  0,  0, 0, 0],
        [0, 0,  0,  0,   0,  0,  0, 0, 0],
        [0, 0,  0,  0,   0,  0,  0, 0, 0],
        [0, 0,  0,  0,   0,  0,  0, 0, 0],
        [0, 0,  0,  0,   0,  0,  0, 0, 0],
        [0, 0,  0, -2,  -5, -2,  0, 0, 0],
        [0, 0,  0, -5, -10, -5,  0, 0, 0],
        [0, 0,  0, -8, -15, -8,  0, 0, 0],
    ],

    'A': [  # Advisor
        [0, 0,  0, -3, -5, -3,  0, 0, 0],
        [0, 0,  0, -2, -3, -2,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0, -2, -3, -2,  0, 0, 0],
        [0, 0,  0, -3, -5, -3,  0, 0, 0],
        [0, 0,  0, -4, -6, -4,  0, 0, 0],
    ],

    'E': [  # Elephant
        [0, 0,  0, -1, -2, -1,  0, 0, 0],
        [0, 0,  0,  2,  2,  2,  0, 0, 0],
        [0, 0,  0,  2,  4,  2,  0, 0, 0],
        [0, 0,  0,  2,  3,  2,  0, 0, 0],
        [0, 0,  0,  1,  1,  1,  0, 0, 0],
        [0, 0,  0,  1,  1,  1,  0, 0, 0],
        [0, 0,  0,  0,  1,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0, -1, -1, -1,  0, 0, 0],
        [0, 0,  0, -2, -2, -2,  0, 0, 0],
    ],

    'R': [  # Rook
        [0, 0,  2,  3,  3,  3,  2, 0, 0],
        [0, 0,  2,  3,  4,  3,  2, 0, 0],
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  1,  2,  2,  2,  1, 0, 0],
        [0, 0,  1,  1,  2,  1,  1, 0, 0],
        [0, 0,  1,  1,  2,  1,  1, 0, 0],
        [0, 0,  2,  3,  3,  3,  2, 0, 0],
        [0, 0,  2,  4,  4,  4,  2, 0, 0],
        [0, 0,  2,  4,  5,  4,  2, 0, 0],
    ],

    'H': [  # Horse
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  2,  4,  5,  4,  2, 0, 0],
        [0, 0,  2,  5,  6,  5,  2, 0, 0],
        [0, 0,  2,  4,  6,  4,  2, 0, 0],
        [0, 0,  1,  3,  4,  3,  1, 0, 0],
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  1,  2,  2,  2,  1, 0, 0],
        [0, 0,  0,  2,  2,  2,  0, 0, 0],
        [0, 0,  0,  1,  1,  1,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
    ],

    'C': [  # Cannon
        [0, 0,  1,  2,  2,  2,  1, 0, 0],
        [0, 0,  1,  3,  4,  3,  1, 0, 0],
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  1,  2,  2,  2,  1, 0, 0],
        [0, 0,  1,  2,  2,  2,  1, 0, 0],
        [0, 0,  0,  1,  2,  1,  0, 0, 0],
        [0, 0,  0,  1,  1,  1,  0, 0, 0],
        [0, 0,  0,  1,  1,  1,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
    ],

    'S': [  # Soldier
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
        [0, 0,  1,  2,  2,  2,  1, 0, 0],
        [0, 0,  1,  3,  4,  3,  1, 0, 0],
        [0, 0,  1,  3,  5,  3,  1, 0, 0],
        [0, 0,  1,  2,  3,  2,  1, 0, 0],
        [0, 0,  1,  1,  2,  1,  1, 0, 0],
        [0, 0,  1,  1,  1,  1,  1, 0, 0],
        [0, 0,  0,  1,  1,  1,  0, 0, 0],
        [0, 0,  0,  0,  1,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0,  0, 0, 0],
    ]
}


def opposite_color(color):
    return 'b' if color == 'r' else 'r'

def generate_threat_map(board):
    threat_map = {'r': set(), 'b': set()}

    black_moves = get_legal_moves(board, 'b')
    red_moves   = get_legal_moves(board, 'r')

    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece != "--":
                color = piece[0]
                moves = black_moves if color == 'b' else red_moves
                piece_attacks = [
                    move for move in moves
                    if move[0] == (r, c) and board[move[1][0]][move[1][1]] != "--"
                ]

                threat_map[color].update(moves)
    return threat_map

def is_connected_soldier(board, r, c, color):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            neighbor = board[nr][nc]
            if neighbor == f"{color}P":
                return True
    return False

def cannon_alignment_score(board, r, c, color):
    score = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        screen_found = False
        nr, nc = r + dr, c + dc
        while 0 <= nr < ROWS and 0 <= nc < COLS:
            target = board[nr][nc]
            if target != "--":
                if not screen_found:
                    screen_found = True
                else:
                    if target[0] != color:
                        score += PIECE_VALUES[target[1]] * 0.3
                    break
            nr += dr
            nc += dc
    return score

def is_passed_soldier(board, r, c, color):
    direction = -1 if color == 'r' else 1
    for i in range(r + direction, 10 if color == 'r' else -1, direction):
        if board[i][c] == f"{opposite_color(color)}P":
            return False
    return True

def evaluate_board(board, turn):
    score = 0
    threat_map = generate_threat_map(board)

    legal_moves = get_legal_moves(board, turn)

    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece == "--":
                continue;

            color, ptype = piece[0], piece[1]
            base_value = PIECE_VALUES[ptype]

            # Positional bonus
            pos_bonus = POSITION_TABLES[ptype][r][c] if color == 'r' else POSITION_TABLES[ptype][9 - r][c]

            # Center control
            center_bonus = 3 if (r, c) in CENTER_CONTROL else 0

            # Advancement bonus (non-linear, via POSITION_TABLES)
            advancement_bonus = 0
            if ptype in ADVANCEMENT_BONUS:
                advancement = r if color == 'b' else 9 - r
                advancement_bonus = ADVANCEMENT_BONUS[ptype] * advancement

            # King safety: penalize king outside palace
            king_safety_penalty = 0
            if ptype == 'K' and not in_palace(r, c, color):
                king_safety_penalty = -20

            # Mobility (number of legal moves for this piece)
            piece_moves = [move for move in legal_moves if move[0] == (r, c)]
            mobility = len(piece_moves) * 0.2

            # Threat penalty / attacker bonus
            threat_penalty = 0
            if (r, c) in threat_map[color]:
                threat_penalty = -base_value * 0.5  # Penalize being under threat
            if (r, c) in threat_map[opposite_color(color)]:
                threat_penalty += base_value * 0.1  # Bonus for attacking

            # Soldier structure bonus
            structure_bonus = 0
            if ptype == 'S':
                if is_connected_soldier(board, r, c, color):
                    structure_bonus += 5
                if is_passed_soldier(board, r, c, color):
                    structure_bonus += 8

            # Cannon alignment bonus
            cannon_bonus = 0
            if ptype == 'C':
                cannon_bonus += cannon_alignment_score(board, r, c, color)

            total = (
                base_value + pos_bonus + center_bonus +
                advancement_bonus + king_safety_penalty +
                mobility + threat_penalty +
                structure_bonus + cannon_bonus
            )

            score += total if color == turn else -total

    return score


def score_move(move, board, turn):
    score = 0

    # Capture scoring (MVV-LVA)
    sr, sc = move[0]
    er, ec = move[1]

    piece = board[sr][sc]
    target = board[er][ec]
    if target != '--':
        attacker_value = PIECE_VALUES[piece[1]]
        victim_value = PIECE_VALUES[target[1]]
        score += 100_000 + (victim_value * 10 - attacker_value)

    return score

def order_moves(moves, board, turn):
    move_scores = []
    for move in moves:
        move_scores.append((score_move(move, board, turn), move))
    move_scores.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in move_scores]


nodes_searched = 0

def quiescence_search(board, turn, alpha, beta):
    stand_pat = evaluate_board(board, turn)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in get_legal_moves(board, turn):
        sr, sc = move[0]
        er, ec = move[1]
        if board[er][ec] != "--":
            new_board = make_move(board, move)
            score = -quiescence_search(new_board, opposite_color(turn), -beta, -alpha)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

def negamax(board, depth, alpha, beta, turn):
    global nodes_searched
    nodes_searched += 1

    if is_checkmate(board, turn):
        return -10000, None

    key = compute_zobrist_hash(board)
    if key in HASH_CACHE and HASH_CACHE[key]['depth'] >= depth:
        return HASH_CACHE[key]['value'], HASH_CACHE[key]['move']

    if depth == 0:
        val = quiescence_search(board, turn, alpha, beta)
        return val, None

    moves = get_legal_moves(board, turn)
    ordered_moves = order_moves(moves, board, turn)

    max_eval = float('-inf')
    best_move = None
    for move in ordered_moves:
        new_board = make_move(board, move)
        eval, _ = negamax(new_board, depth - 1, -beta, -alpha, opposite_color(turn))
        eval = -eval
        if eval > max_eval:
            max_eval = eval

            best_move = move
        alpha = max(alpha, eval)
        if alpha >= beta:
            break

    HASH_CACHE[key] = {'value': max_eval, 'move': best_move, 'depth': depth}
    return max_eval, best_move

def iterative_deepening(board, max_depth, turn):
    result = {'move': None}
    def search():
        for depth in range(1, max_depth + 1):
            eval, move = negamax(board, depth, float('-inf'), float('inf'), turn)
            if move:
                result['move'] = move

            global nodes_searched
            print(f"- Best move at depth {depth}: {move[0]} -> {move[1]} | Score: {eval:.2f}")
            print(f"- Total positions searched at depth {depth}: {nodes_searched}")
            nodes_searched = 0

    thread = threading.Thread(target=search)
    thread.start()
    thread.join(timeout=None)
    return result['move']




def main():
    # Pass in the color you want to play as (RED or BLACK)
    env = XiangQiEnv(BLACK)
    env.render()
    agent = RandomAgent()
    board = copy.deepcopy(INITIAL_BOARD)
    OPENING_LIMIT = 2
    done = False
    turn_number = 0
    round = 0
    obs = None
    while not done:
        if env.turn == ALLY:
            # print(obs)
            
            board = convert_chessboard(obs)
            symbolic_board = flip_chessboard_properly(board)
            # print(symbolic_board)
           
            # for row in symbolic_board:
            #     print(row)
            ai_move = None
            if turn_number < OPENING_LIMIT:
                ai_move = get_opening_move(symbolic_board)
            if not ai_move:
                ai_move = iterative_deepening(symbolic_board, 2, 'b')
            
            turn_number += 1
            print(ai_move)
            print(flip_move(ai_move))
            

            ai_move = flip_move(ai_move)
            piece_at_start = get_piece_at_position(obs, ai_move[0])
            move_ai = move_to_action_space( piece_at_start ,ai_move[0], ai_move[1])
            obs, reward, done, _ = env.step(move_ai)
            if(done):
                print("AI Win")

            player = "You"
            # print(piece)
            start = ai_move[0]
            end = ai_move[1]
            piece = PIECE_ID_TO_NAME[piece_at_start]
        else:
            time.sleep(1)
            action = agent.move(env)
            obs, reward, done, _ = env.step(action)
            if(done):
                print("Random Agent Win")

            player = "RL Agent"
            move = action_space_to_move(action)
            print(move[0])
            piece = PIECE_ID_TO_NAME[move[0]]
            start = move[1]
            end = move[2]

        env.render()
        round += 1
        print(f"Round: {round}")
        print(f"{player} made the move {piece} from {start} to {end}.")
        print(f"Reward: {reward}")
        print("================")
    
    print("Closing Xiangqi environment")
    env.close()


if __name__ == '__main__':
    main()



