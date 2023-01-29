def get_moves(board, i, j):
    moves = 0
    # Check up
    if i > 1 and board[i - 1][j] == "o" and board[i - 2][j] == ".":
        moves +=1

    # Check down
    if i < 7 and board[i + 1][j] == "o" and board[i + 2][j] == ".":
        moves +=1

    # Check left
    if j > 1 and board[i][j - 1] == "o" and board[i][j - 2] == ".":
        moves += 1

    # Check right
    if j < 7 and board[i][j + 1] == "o" and board[i][j + 2] == ".":
        moves += 1

    # Return number of moves
    return moves


# Create board with padding
board = []
board.append([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ])
for i in range(1, 8):
    board.append([])
    board[i].append(' ')
    for c in input():
        board[i].append(c)
    board[i].append(' ')
board.append([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ])

# Check for valid moves
moves = 0
for i in range(1, 8):
    for j in range(1, 8):
        # Check if board[i][j] has a valid move
        if board[i][j] == "o":
            moves += get_moves(board, i, j)

# Output number of moves
print(moves)
