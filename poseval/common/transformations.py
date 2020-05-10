import numpy as np


class MalformedFenStringError(Exception):
    pass


class AlreadyWhiteError(Exception):
    pass


class DataSpecs:
    tensor6x8x8_sparse = 'tensor6x8x8'
    vector6x8x8_flat = 'vector6x8x8'
    tensor12x8x8_sparse = 'tensor12x8x8'
    vector12x8x8_flat = 'vector12x8x8'

    piece2layer_for_6x8x8_tensor = {
        'p': 0, 'P': 0, 'n': 1, 'N': 1, 'b': 2, 'B': 2,
        'r': 3, 'R': 3, 'q': 4, 'Q': 4, 'k': 5, 'K': 5
    }

    piece2value_for_6x8x8_tensor = {
        'p': 1, 'P': -1, 'n': 1, 'N': -1, 'b': 1, 'B': -1,
        'r': 1, 'R': -1, 'q': 1, 'Q': -1, 'k': 1, 'K': -1
    }

    piece2layer_for_12x8x8_tensor = {
        'p': 0, 'P': 1, 'n': 2, 'N': 3, 'b': 4, 'B': 5,
        'r': 6, 'R': 7, 'q': 8, 'Q': 9, 'k': 10, 'K': 11
    }

    piece2value_for_12x8x8_tensor = {
        'p': 1, 'P': 1, 'n': 1, 'N': 1, 'b': 1, 'B': 1,
        'r': 1, 'R': 1, 'q': 1, 'Q': 1, 'k': 1, 'K': 1
    }

    col_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    @staticmethod
    def string_square_to_8x8(str_square: str):
        return int(str_square[1]) - 1, DataSpecs.col_list.index(str_square[0])


class Fen:
    def __init__(self, fenstring):
        self.elems = fenstring.split(" ")
        self.fenstring = fenstring
        self.validate_fenstring()

    def validate_fenstring(self):
        if len(self.elems) != 6:
            raise MalformedFenStringError

    def can_castle_white_king(self):
        return 'K' in self.elems[2]

    def can_castle_black_king(self):
        return 'k' in self.elems[2]

    def can_castle_black_queen(self):
        return 'q' in self.elems[2]

    def can_castle_white_queen(self):
        return 'Q' in self.elems[2]

    def next_to_move(self):
        return self.elems[1]

    def castling_vector(self):
        return [
            self.can_castle_black_king(),
            self.can_castle_white_king(),
            self.can_castle_black_queen(),
            self.can_castle_white_queen(),
        ]

    def en_passant_target_square(self):
        return None if self.elems[3] == '-' else self.elems[3]

    def transform(self, transformation_type=DataSpecs.vector6x8x8_flat):
        if transformation_type == DataSpecs.vector6x8x8_flat:
            return self.raw_board_to_6x8x8_flat_vector()
        elif transformation_type == DataSpecs.vector12x8x8_flat:
            return self.raw_board_to_12x8x8_flat_vector()
        elif transformation_type == DataSpecs.tensor6x8x8_sparse:
            return self.raw_board_to_6x8x8_sparse_representation()
        elif transformation_type == DataSpecs.tensor12x8x8_sparse:
            return self.raw_board_to_12x8x8_sparse_representation()
        else:
            raise ValueError("No such transformation type")

    def raw_board(self):
        return self.elems[0]

    def raw_board_to_6x8x8_flat_vector(self):
        return self._raw_board_to_flat_vector(
            DataSpecs.piece2layer_for_6x8x8_tensor, DataSpecs.piece2value_for_6x8x8_tensor, 6
        )

    def raw_board_to_12x8x8_flat_vector(self):
        return self._raw_board_to_flat_vector(
            DataSpecs.piece2layer_for_12x8x8_tensor, DataSpecs.piece2value_for_12x8x8_tensor, 12
        )

    def raw_board_to_6x8x8_sparse_representation(self):
        return self._raw_board_to_sparse_representation(
            DataSpecs.piece2layer_for_6x8x8_tensor, DataSpecs.piece2value_for_6x8x8_tensor, 6
        )

    def raw_board_to_12x8x8_sparse_representation(self):
        return self._raw_board_to_sparse_representation(
            DataSpecs.piece2layer_for_12x8x8_tensor, DataSpecs.piece2value_for_12x8x8_tensor, 12
        )

    def flip(self):
        self.elems[1] = 'w' if self.elems[1] == 'b' else 'b'
        self.flip_position()
        self.flip_castling()
        self.flip_en_passant()
        self.fenstring = " ".join(self.elems)

    def flip_position(self):
        self.elems[0] = "/".join(
            ''.join([self._reverse_piece(piece) for piece in line]) for line in self.elems[0].split("/")[::-1])

    def flip_castling(self):
        if self.elems[2] == '-':
            pass
        else:
            new_castling_string = ''
            for el in 'kqKQ':
                if el in self.elems[2]:
                    new_castling_string += self._reverse_piece(el)
            self.elems[2] = new_castling_string

    def flip_en_passant(self):
        if self.elems[3] == '-':
            pass
        else:  # otherwise, it's a square
            curr_column, curr_line = self.elems[3]
            new_col = 8 - DataSpecs.col_list.index(curr_column)
            new_line = 8 - int(curr_line) + 1  # add 1, because line numbers start at 1
            self.elems[3] = DataSpecs.col_list[new_col] + str(new_line)

    @staticmethod
    def reverse_square(square):
        return

    @staticmethod
    def _reverse_piece(fen_el):
        if not isinstance(fen_el, str):
            return fen_el
        elif fen_el.islower():  # it's a black piece make it white
            return fen_el.upper()
        else:  # piece.isupper(), so it's a white piece, make it black
            return fen_el.lower()

    def _raw_board_to_sparse_representation(self, piece_to_layer_map, piece_to_value_map, layers):
        raw_board = self.raw_board()
        row = 0
        column = 0
        indices_x = []
        indices_y = []
        indices_z = []
        values = []
        for c in raw_board:
            if c.isdigit():
                column = column + int(c)
            elif c == '/':
                row = row + 1
                column = 0
            else:
                indices_z.append(piece_to_layer_map[c])
                indices_x.append(row)
                indices_y.append(column)
                values.append(piece_to_value_map[c])
                column = column + 1

        return {'indices_z': indices_z, 'indices_x': indices_x, 'indices_y': indices_y, 'values': values}

    def _raw_board_to_flat_vector(self, piece_to_layer_map, piece_to_value_map, layers):
        raw_board = self.raw_board()
        row = 0
        column = 0
        vector = [0] * (8 * 8 * layers)
        for c in raw_board:
            if c.isdigit():
                column = column + int(c)
            elif c == '/':
                row = row + 1
                column = 0
            else:
                vector[layers * (8 * row + column) + piece_to_layer_map[c]] = piece_to_value_map[c]
                column = column + 1
        return vector

    def castling_vector_8x8x2(self):
        board = np.zeros([8, 8, 2])
        board[(*DataSpecs.string_square_to_8x8('g8'), 1)] = self.castling_vector()[0]
        board[(*DataSpecs.string_square_to_8x8('g1'), 0)] = self.castling_vector()[1]
        board[(*DataSpecs.string_square_to_8x8('c8'), 1)] = self.castling_vector()[2]
        board[(*DataSpecs.string_square_to_8x8('c1'), 0)] = self.castling_vector()[3]
        return board

    def en_passant_vector_8x8x1(self):
        board = np.zeros([8, 8, 1])
        if self.en_passant_target_square():
            board[(*DataSpecs.string_square_to_8x8(self.en_passant_target_square()), 0)] = 1
        return board

    def full_board_representation(self, dim):
        if dim == 6:
            raise NotImplementedError  # TODO
        return np.dstack(
            (self.pieces_dense_representation(dim),
             self.castling_vector_8x8x2(),
             self.en_passant_vector_8x8x1()))

    def pieces_dense_representation(self, dim):
        board = np.zeros([8, 8, dim])
        if dim == 6:
            index_dict = self.raw_board_to_6x8x8_sparse_representation()
        elif dim == 12:
            index_dict = self.raw_board_to_12x8x8_sparse_representation()
        else:
            raise ValueError('Dimension not supported')
        coordinates_list = zip(index_dict['indices_x'], index_dict['indices_y'], index_dict['indices_z'])
        for coordinates, value in zip(coordinates_list, index_dict['values']):
            board[coordinates] = value
        return board
