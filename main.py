import chess
import chess.engine
import os
import tomllib
from tqdm import tqdm

cfg = tomllib.load(open("lc0-config.toml", "rb"))

ENGINE_PATH = cfg["General"]["engine_path"]
SEARCH_LIMITS = cfg["SearchLimits"]
# FILENAME = "data/fens_50k_synthetic_positions_synthetic_positions_with_no_pawns_found_by_evolutionary_algorithm.csv"


def rotate_90(board):
    return chess.flip_vertical(chess.flip_diagonal(board))


def rotate_180(board):
    return rotate_90(rotate_90(board))


def rotate_270(board):
    return rotate_180(rotate_90(board))


def start_engine(cfg):
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
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


def forced_move_positions(filename="chess_input_data/fens_400k_forced_move_positions.csv"):
    engine = start_engine(cfg)

    basename = os.path.basename(filename)
    out_filename = f"results/results_{basename}"

    results = []

    with open(filename, "r") as f:
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

            # TODO Check if this is the same score calculation as in the paper
            score1 = info1["score"].wdl().relative.expectation()
            score2 = info2["score"].wdl().relative.expectation()
            score_diff = score2 - score1

            print(score1, score2)

            results.append((fen1, fen2, move, score1, score2, score_diff))

    with open(out_filename, "w") as f:
        f.write("fen1,fen2,move,score1,score2,difference")
        for fen1, fen2, move, score1, score2, score_diff in results:
            f.write(f"{fen1},{fen2},{move},{score1},{score2},{score_diff}\n")

    engine.quit()


def evaluate_position(engine, fen):
    board = chess.Board(fen)
    # TODO Figure out how to use nodes as search limits, not only time
    info = engine.analyse(board, chess.engine.Limit(time=0.1))  # **SEARCH_LIMITS))
    return info["score"]


if __name__ == "__main__":
    forced_move_positions()
