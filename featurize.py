import numpy as np
from game_info import RESOURCES, QUESTS, QUEST_TYPES, DEFAULT_BUILDINGS, Quest
from player import Player
from board import BoardState
from game import GameState

def featurizeResources(resources: dict[str, int], includeVP: bool = False, includeQ: bool = False):
    '''Featurize a resource vector.'''
    # TODO (later): change "includeQVP" to "includeQI"
    possibleResources = RESOURCES.copy() # All resources, incluidng VP, I, and Q
    if not includeVP:
        possibleResources.remove("VP")
    if not includeQ:
        possibleResources.remove("Q")

    # Each element of the feature vector corresponds to one resource type
    return np.array([resources.get(resource, 0) for resource in possibleResources])

def featurizeQuest(quest: Quest):
    '''Featurize a quest.'''
    # One-hot vector for quest type
    typeVector = (np.array(QUEST_TYPES) == quest.type).astype(int)

    # Appent feature vectors for requirements and rewards
    return np.hstack([typeVector, 
                      featurizeResources(quest.requirements), 
                      featurizeResources(quest.rewards, includeVP=True, includeQ=True)])

# The length of a quest feature vector (for use in zero-blocks)
QUEST_FEATURE_LEN = len(featurizeQuest(QUESTS[0]))

# Define the maximum number of quests a player can have.
# This determines the number of quest feature vector blocks 
# there is room for in the overall feature vectors.
MAX_QUESTS = 15

def featurizePlayer(player: Player):
    '''Featurize a player state.'''
    # One element for number of agents
    agentsVec = np.array([player.agents,])

    # Vector of the player's resources
    resourceVec = featurizeResources(player.resources, includeVP=True)

    # TODO (later): Add number of Intrigue cards and plot quests

    # Vector of the player's featurized active/completed quests
    questFeaturesList = []
    for playerQuests in [player.activeQuests, player.completedQuests]:
        for i in range(MAX_QUESTS):
            if i < len(playerQuests):
                questFeaturesList.append(featurizeQuest(playerQuests[i]))
            else:
                questFeaturesList.append(np.zeros(QUEST_FEATURE_LEN))

    return np.hstack([agentsVec, resourceVec] + questFeaturesList)


# Featurize game state
# 
# (Note for self later: instead of including players always in the same order and the turn order separately, 
# possibly just include the other players in turn order in the game state inherently)

def featurizeGameState(gameState: GameState):
    boardState = gameState.boardState
    playerNames = gameState.playerNames
    # TODO: num rounds left? anything else?

    # Concatendated one-hot vectors for building occupations by player
    buildingStateVec = []
    for building in DEFAULT_BUILDINGS:
        oneBuildingStateVec = np.array(playerNames) == boardState.buildingStates[building]
        buildingStateVec.append(int())
    # TODO: Chang ebuilding states to be one-hot vectors for occupations of each player instead of just 0/1 occupied/unoccupied

    # TODO (later): do the same but for all possible building spots

    # TODO: put the four featurized cliffwatch quests here
    # TODO: featurize players

# Note for self later: Although one large CNN would not work, consider forcing 
# the first layer to be the same for each quest block, for each player block, etc.

# TODO: Write test functions which print the game state and actions 
# in words to compare with the displayed game state before featurization

def main():
    # Test the quest featurization
    quest = QUESTS[np.random.randint(len(QUESTS))]
    print(quest, featurizeQuest(quest))