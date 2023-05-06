from random import shuffle
from game_info import Quest, LORD_CARDS, agentsPerPlayer
from player import Player
from game_state import GameState

# Class to control the flow of the game, focused on turn progression and move 
# options. Broadly, this class handles anything involving the game state and
# the players, while those other classes handle either only the game state
# itself or only the players themselves (to the extent possible).

class GameControl():
    '''
    Class to control the flow of the game, 
    focused on turn progression and move 
    options. 
    '''
    def __init__(self, numPlayers: int = 3, numRounds: int = 8, 
                 playerNames = None):
        '''
        Initialize the game state and players.

        Args: 
            numPlayers: the number of players in the game
            numRounds: the number of rounds in the game
            playerNames (optional): the names for each player
        '''
        # Initialize the remaining number of rounds
        self.roundsLeft = numRounds

        # Initialize the GameState
        self.gameState = GameState()

        # Check that we have a valid number of players
        assert numPlayers >= 2 and numPlayers <= 5
        self.numPlayers = numPlayers

        # Set default player names
        self.playerNames = playerNames
        if playerNames == None:
            self.playerNames = [
                "PlayerOne", "PlayerTwo", 
                "PlayerThree", "PlayerFour",
                "PlayerFive"
            ][:numPlayers]
        
        # Shuffle the lord cards
        shuffled_lord_cards = LORD_CARDS.copy()
        print(shuffled_lord_cards[:5])
        shuffle(shuffled_lord_cards)
        print(shuffled_lord_cards[:5])

        # Initialize the players
        self.players = []
        for i in range(numPlayers):
            print(shuffled_lord_cards[i])
            self.players.append(Player(self.playerNames[i], agentsPerPlayer(numPlayers),
                                       shuffled_lord_cards[i]))

        # Deal quest cards to players
        for _ in range(2):
            for player in self.players:
                player.getQuest(self.gameState.drawQuest())

        # TODO (later version): Deal intrigue cards to players

        # Define the turn order 
        shuffled_indices = list(range(1,numPlayers+1))
        shuffle(shuffled_indices)
        self.turnOrder = {shuffled_indices[i]: player for i,player in enumerate(self.players)}

        # Set the current place in the turn order (1,...,numPlayers)
        self.currentTurn = 1

        # TODO: Deal gold to players

        # Finally, start a new round (at this stage, just 
        # decrements roundsLeft and places VPs on
        # buildings at builder's hall)
        self.newRound()

    def newRound(self):
        '''Reset the board at the beginning of each round.'''
        self.roundsLeft -= 1
        # TODO (later version): put VPs on buildings at bulider's hall

        # Reset all buildings
        self.gameState.clearBuildings()

        # TODO (later version): put new resources on buildings that need them

        # Get new agent at fifth round
        if self.roundsLeft == 4:
            for player in self.players:
                player.getAgent()

        # Return all agents
        for player in self.players:
            player.returnAgents()

    def takeTurn(self):
        '''Take a single turn in the turn order.'''
        # TODO: Implement 
        currentPlayer = self.turnOrder[self.currentTurn]
        possibleMoves = [] # Fill this 
        move = currentPlayer.selectMove(possibleMoves) # Implement this
        # Execute this move. Maybe have a different function for 
        # selecting a building to play at vs other sub-selections
        # such as a quest to complete or an intrigue card to play?
        self.turnOrder += 1 # TODO: Should be order = order + 1 mod nplayers?

    def runGame(self):
        '''Umbrella function to run the game.'''
        while self.roundsLeft > 0:
            # TODO: Make sure this accounts for multiple agents for player
            while self.turnOrder <= self.numPlayers:
                self.takeTurn()
            self.newRound()

    def displayGame(self) -> None:
        '''Display the state of the game and players.'''
        # TODO: Implement this to display the state of the game
        # (I moved this from game_state to here so it can 
        # display the player states as well)
        # Will start textually, should (maybe) be graphically later (if we have time)
        pass
        

def main():
    # Test gameControl
    gameControl = GameControl(numPlayers=2)
    print(gameControl.playerNames)
    for i,player in gameControl.turnOrder.items():
        print(i,player.name, player.lordCard)