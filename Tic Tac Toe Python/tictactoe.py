# initialize game
# create the board, the current player and winner statuses
import random 

board = [
    '-', '-', '-',
    '-', '-', '-',
    '-', '-', '-'
]

currentPlayer = 'X'
winner = None
gameRunning = True

# function to print the board
def printBoard(board):
    print(board[0] + ' | ' + board[1] + ' | ' + board[2])
    print('----------')
    print(board[3] + ' | ' + board[4] + ' | ' + board[5])
    print('----------')
    print(board[6] + ' | ' + board[7] + ' | ' + board[8])

#printBoard(board)

# take player input
def playerInput(board):
    inp = int(input('Enter a number 1 - 9: '))
    # validation checks
    if inp >= 1 and inp <= 9 and board[inp-1] == '-':
        board[inp-1] = currentPlayer
    else:
        print('Oops another player is already in that spot')

# check for win or tie
# tic tac toe can be won with horizontal, vertical or diagonal lines
def checkHorizontal(board):
    global winner
    if board[0] ==  board[1] == board[2] and board[0] != '-':
         winner = board[0]
         return True
    elif board[3] ==  board[4] == board[5] and board[3] != '-':
         winner = board[3]
         return True
    elif board[6] ==  board[7] == board[8] and board[6] != '-':
         winner = board[6]
         return True
    
def checkVertical(board):
    global winner
    if board[0] ==  board[3] == board[6] and board[0] != '-':
         winner = board[0]
         return True
    elif board[1] ==  board[4] == board[7] and board[1] != '-':
         winner = board[1]
         return True
    elif board[2] ==  board[5] == board[8] and board[2] != '-':
         winner = board[2]
         return True
    
def checkDiagonal(board):
    global winner
    if board[0] ==  board[4] == board[8] and board[0] != '-':
         winner = board[0]
         return True
    elif board[2] ==  board[4] == board[6] and board[2] != '-':
         winner = board[2]
         return True

def checkWin(board):
    global gameRunning
    if checkHorizontal(board):
        printBoard(board)
        print('The winner is {}'.format(winner))
        gameRunning = False
    elif checkVertical(board):
        printBoard(board)
        print('The winner is {}'.format(winner))
        gameRunning = False
    elif checkDiagonal(board):
        printBoard(board)
        print('The winner is {}'.format(winner))
        gameRunning = False

# check for a tie
def checkTie(board):
    global gameRunning
    if '-' not in board:
        printBoard(board)
        print('Tie Game!')
        gameRunning = False

# switch player
def switchPlayer():
    # no argument needed as we will not be modifiying the board
    global currentPlayer
    if currentPlayer == 'X':
        currentPlayer = 'O'
    else:
        currentPlayer = 'X'

# include a computer playing with random guesses
def computer(board):
    while currentPlayer == "O":
        position = random.randint(0, 8)
        if board[position] == "-":
            board[position] = "O"
            switchPlayer()

# game loop
while gameRunning:
    printBoard(board)
    playerInput(board)
    checkWin(board)
    checkTie(board)
    switchPlayer()
    computer(board)
    checkWin(board)
    checkTie(board)

