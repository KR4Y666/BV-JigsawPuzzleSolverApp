import math

def calculate_piece_size(image, number_of_pieces):
    width, height = image.size
    area = width * height
    piece_area = area / number_of_pieces
    piece_width = int(math.sqrt(piece_area))
    piece_height = int(math.sqrt(piece_area))

    return piece_width, piece_height
