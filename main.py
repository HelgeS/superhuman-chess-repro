import chess
import chess.engine
import os
import tomllib
from tqdm import tqdm
import numpy as np


# These functions come from lc0
# Where the constants come from is unclear
# There are different constants in earlier versions.
# The superhuman original uses the same functions for conversion.
def cp2q(cp):
    """
    Convert centipawn score to q value used for analysis.
    """
    return np.arctan(cp / 90) / 1.5637541897


def q2cp(q):
    """
    Convert q value used for analysis to centipawn score.
    """
    return 90 * np.tan(1.5637541897 * q)




def calculate_scores(score_info):
    """
    Calculate q and win probability from engine analysis info.
    
    q is used for analysis.
    win_probability is used for plots.
    """
    wdl = score_info.wdl().relative
    p_win = wdl.winning_chance()
    p_loss = wdl.losing_chance()
    p_draw = wdl.drawing_chance()
    q = p_win - p_loss
    win_probability = (q + 1 - p_draw) / 2
    return q, win_probability


def rotate_90(board):
    return chess.flip_vertical(chess.flip_diagonal(board))


def rotate_180(board):
    return rotate_90(rotate_90(board))


def rotate_270(board):
    return rotate_180(rotate_90(board))


def start_engine(cfg):
    engine = chess.engine.SimpleEngine.popen_uci(cfg["General"]["engine_path"])
    engine.configure(
        {
            "WeightsFile": cfg["General"]["network_base_path"],
            "Backend": cfg["EngineConfig"]["Backend"],
            "VerboseMoveStats": cfg["EngineConfig"]["VerboseMoveStats"],
            "SmartPruningFactor": cfg["EngineConfig"]["SmartPruningFactor"],
            "Threads": cfg["EngineConfig"]["Threads"],
            "TaskWorkers": cfg["EngineConfig"]["TaskWorkers"],
            "MinibatchSize": cfg["EngineConfig"]["MinibatchSize"],
            "MaxPrefetch": cfg["EngineConfig"]["MaxPrefetch"],
            "NNCacheSize": cfg["EngineConfig"]["NNCacheSize"],
            "TwoFoldDraws": cfg["EngineConfig"]["TwoFoldDraws"],
        }
    )
    return engine


# engine = start_engine(cfg)


def position_mirroring(
    engine,
    cfg,
    filename,
):
    SEARCH_LIMITS = cfg["SearchLimits"]

    basename = os.path.basename(filename)
    out_filename = f"results/results_{basename}"

    results = []

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines[:1000]):
            fen = line.strip()

            board = chess.Board(fen)

            info = engine.analyse(board, chess.engine.Limit(**SEARCH_LIMITS))
            q_original, _ = calculate_scores(info["score"])

            board = board.mirror()

            info = engine.analyse(board, chess.engine.Limit(**SEARCH_LIMITS))
            q_mirror, _ = calculate_scores(info["score"])

            # results should be equal
            difference = abs(q_original - q_mirror)

            results.append((board.fen(), q_original, q_mirror, difference))

    with open(out_filename, "w") as f:
        f.write("fen,original,mirror,difference\n")
        for fen, original, mirror, difference in results:
            f.write(f"{fen},{original},{mirror},{difference}\n")


def board_transformations(
    engine,
    cfg,
    filename,
):
    transformations = [
        "original",
        "rot90",
        "rot180",
        "rot270",
        "flip_diag",
        "flip_anti_diag",
        "flip_hor",
        "flip_vert",
    ]

    SEARCH_LIMITS = cfg["SearchLimits"]

    basename = os.path.basename(filename)
    out_filename = f"results/results_{basename}"

    results = []

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines[:2000]):
            fen = line.strip()

            board = chess.Board(fen)

            tf_scores = []

            for tf in transformations:
                if tf == "original":
                    board_tf = board.copy()
                elif tf == "rot90":
                    board_tf = board.transform(rotate_90)
                elif tf == "rot180":
                    board_tf = board.transform(rotate_180)
                elif tf == "rot270":
                    board_tf = board.transform(rotate_270)
                elif tf == "flip_diag":
                    board_tf = board.transform(chess.flip_diagonal)
                elif tf == "flip_anti_diag":
                    board_tf = board.transform(chess.flip_anti_diagonal)
                elif tf == "flip_hor":
                    board_tf = board.transform(chess.flip_horizontal)
                elif tf == "flip_vert":
                    board_tf = board.transform(chess.flip_vertical)
                else:
                    raise ValueError(f"Unknown transformation: {tf}")

                info_tf = engine.analyse(board_tf, chess.engine.Limit(**SEARCH_LIMITS))
                q, _ = calculate_scores(info_tf["score"])
                tf_scores.append(q)

            # Scores are expected to be the same for all transformations
            difference = max(tf_scores[1:]) - tf_scores[0]
            tf_scores.append(difference)

            print(fen, difference)

            results.append((board.fen(), tf_scores))

    with open(out_filename, "w") as f:
        # difference = max. difference over all transformations
        f.write(
            "fen,original,rot90,rot180,rot270,flip_diag,flip_anti_diag,flip_hor,flip_vert,difference\n"
        )
        for fen, scores in results:
            score_str = ",".join(map(str, scores))
            f.write(f"{fen},{score_str}\n")


def forced_move_positions(
    engine, cfg, filename="chess_input_data/fens_400k_forced_move_positions.csv"
):
    SEARCH_LIMITS = cfg["SearchLimits"]

    basename = os.path.basename(filename)
    out_filename = f"results/results_{basename}"

    with open(filename, "r") as f, open(out_filename, "w") as fout:
        fout.write("fen1,fen2,move,score1,score2,difference,score1conv,score2conv,differenceconv\n")
        lines = f.readlines()
        for line in tqdm(lines[:2]):
            fen = line.strip()

            board = chess.Board(fen)

            assert board.legal_moves.count() == 1, (
                f"Expected exactly one legal move in position: {fen}"
            )

            fen1 = board.fen()
            info1 = engine.analyse(board, chess.engine.Limit(**SEARCH_LIMITS))

            move = next(board.legal_moves.__iter__())
            board.push(move)

            fen2 = board.fen()
            info2 = engine.analyse(board, chess.engine.Limit(**SEARCH_LIMITS))

            q1, _ = calculate_scores(info1["score"])
            q2, _ = calculate_scores(info2["score"])

            q1conv = cp2q(info1["score"].relative.score(mate_score=12780))
            q2conv = cp2q(info2["score"].relative.score(mate_score=12780))

            # results should be equal but with inverted sign
            score_diff = abs(q1 + q2)
            score_diff_conv = abs(q1conv + q2conv)

            fout.write(f"{fen1},{fen2},{move},{q1},{q2},{score_diff},{q1conv},{q2conv},{score_diff_conv}\n")

    #         results.append((fen1, fen2, str(move), q1, q2, score_diff, float(q1conv), float(q2conv), float(score_diff_conv)))
    #         # print(results[-1])

    # with 
    #     for fen1, fen2, move, score1, score2, score_diff, score1conv, score2conv, score_diff_conv in results:
            

if __name__ == "__main__":
    cfg = tomllib.load(open("lc0-config.toml", "rb"))

    # FILENAME = "data/fens_50k_synthetic_positions_synthetic_positions_with_no_pawns_found_by_evolutionary_algorithm.csv"

    engine = start_engine(cfg)

    forced_move_positions(
        engine,
        cfg,
        # filename="/home/helge/D1/superhuman/chess_input_data/fens_400k_forced_move_positions.csv",
    )
    # position_mirroring(
    #     engine,
    #     cfg,
    #     filename="/home/helge/D1/superhuman/chess_input_data/fens_400k_middlegame_positions_for_board_mirroring.csv",
    # )
    # board_transformations(
    #     engine,
    #     cfg,
    #     filename="/home/helge/D1/superhuman/chess_input_data/fens_50k_synthetic_positions_with_no_pawns_for_board_transformations.csv",
    # )

    engine.quit()
