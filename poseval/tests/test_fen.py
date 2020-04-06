import numpy as np
import pytest

from poseval.common.transformations import DataSpecs, Fen


def test_if_flipped_twice_produce_original_input():
    flip_twice_and_compare("8/1P4B1/8/1q3k2/2Q5/8/4r1Np/K7 b - - 0 0")
    flip_twice_and_compare("7k/Pn1R4/8/5q2/2K3Q1/8/1b4p1/8 w - - 0 0")
    flip_twice_and_compare("3qk3/8/8/8/8/8/8/3KQ3 b - - 0 0")
    flip_twice_and_compare("4k3/5p2/8/2K5/8/8/8/8 w - - 0 0")
    flip_twice_and_compare("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR b KQkq - 0 0")
    flip_twice_and_compare("rnbq1rk1/p1p1bpp1/1p2pn1p/3p4/2PP3B/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8")
    flip_twice_and_compare("rnbqkbnr/pppp1ppp/4p3/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2")


# https://lichess.org/editor/8/1P4B1/8/1q3k2/2Q5/8/4r1Np/K7_b_-_-
def test_reverse_fen_end_game_pawns_to_be_promoted():
    input_fen = "8/1P4B1/8/1q3k2/2Q5/8/4r1Np/K7 b - - 0 0"
    output_fen = "7k/Pn1R4/8/5q2/2K3Q1/8/1b4p1/8 w - - 0 0"
    reverse_and_compare(input_fen, output_fen)


# https://lichess.org/editor/8/4qk2/5q2/1r1R4/3Q4/2QK4/8/8_b_-_-
def test_reverse_fen_end_game_with_queens_and_rooks():
    input_fen = "8/4qk2/5q2/1r1R4/3Q4/2QK4/8/8 b - - 0 0"
    output_fen = "8/8/4kq2/4q3/4r1R1/2Q5/2KQ4/8 w - - 0 0"
    reverse_and_compare(input_fen, output_fen)


# https://lichess.org/editor/3qk3/8/8/8/8/8/8/3KQ3_b_-_-
def test_reverse_fen_end_game_with_queens():
    input_fen = "3qk3/8/8/8/8/8/8/3KQ3 b - - 0 0"
    output_fen = "3qk3/8/8/8/8/8/8/3KQ3 w - - 0 0"
    reverse_and_compare(input_fen, output_fen)


# https://lichess.org/editor/8/8/8/8/5k2/8/2P5/3K4_b_-_-
def test_reverse_fen_end_game():
    input_fen = "8/8/8/8/5k2/8/2P5/3K4 b - - 0 0"
    output_fen = "4k3/5p2/8/2K5/8/8/8/8 w - - 0 0"
    reverse_and_compare(input_fen, output_fen)


# https://lichess.org/editor/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-
def test_reverse_fen_initial_position():
    input_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"
    output_fen = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR b KQkq - 0 0"
    reverse_and_compare(input_fen, output_fen)


def reverse_and_compare(input_fen, output_fen):
    f = Fen(input_fen)
    f.flip()
    assert f.fenstring == output_fen


def flip_twice_and_compare(input_fen):
    f = Fen(input_fen)
    f.flip()
    f = Fen(f.fenstring)
    f.flip()
    assert f.fenstring == input_fen


def test_string_square_to_8x8():
    assert DataSpecs.string_square_to_8x8('a1') == (0, 0)
    assert DataSpecs.string_square_to_8x8('h8') == (7, 7)
    assert DataSpecs.string_square_to_8x8('e4') == (3, 4)  # unlike regular notation, we invoke lines first
    with pytest.raises(KeyError):
        DataSpecs.string_square_to_8x8('E4')


def test_castling_vector_8x8x2():
    castling_vector = Fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0").castling_vector_8x8x2()
    assert (castling_vector.shape == (8, 8, 2))
    assert np.sum(castling_vector, axis=(0, 1, 2)) == 4.0
    castling_vector = Fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w K - 0 0").castling_vector_8x8x2()
    assert np.sum(castling_vector, axis=(0, 1, 2)) == 1.0
    assert castling_vector[0, 6, 0] == 1.0


def test_en_passant_vector_8x8x1():
    en_p_vec = Fen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1').en_passant_vector_8x8x1()
    assert (en_p_vec.shape == (8, 8, 1))
    assert np.sum(en_p_vec, axis=(0, 1, 2)) == 1
    en_p_vec = Fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0").en_passant_vector_8x8x1()
    assert np.sum(en_p_vec, axis=(0, 1, 2)) == 0


def test_full_board_representation():
    fen = Fen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')
    board = fen.full_board_representation(6)
    assert (board.shape == (8, 8, 9))
    np.testing.assert_array_equal(board[:, :, 8], fen.en_passant_vector_8x8x1().reshape(8, 8), verbose=True)
    np.testing.assert_array_equal(board[:, :, 6:8], fen.castling_vector_8x8x2())
    np.testing.assert_array_equal(board[:, :, :6], fen.pieces_dense_representation(6))
    board = Fen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1').full_board_representation(12)
    assert (board.shape == (8, 8, 15))


def test_sparse_to_dense(self, dim):
    pass


def test_flip_color(self, full_representation):
    pass
