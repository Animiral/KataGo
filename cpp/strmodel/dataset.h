#ifndef STRMODEL_DATASET_H
#define STRMODEL_DATASET_H

#include <map>
#include <string>
#include <vector>
#include "core/logger.h"
#include "neuralnet/strengthnet.h"

// The dataset is a chronological sequence of games with move features.
class Dataset {

public:

  // prediction data to be computed by strength model based on recent moves
  struct Prediction {
    float whiteRating;
    float blackRating;
    float score;
  };

  // data on one game from the dataset list file
  struct Game {
    std::string sgfPath;
    struct {
      std::size_t player; // index of player (given as name string in CSV file)
      float rating;       // target provided in input file
      int prevGame;       // index of most recent game with this player before this or -1
      std::vector<MoveFeatures> features; // precomputed from the moves of this player in this game
    } white, black;
    float score;          // game outcome for black: 0 for loss, 1 for win, 0.5 for undecided
    Prediction prediction;
    
    enum {
      none = 0,
      training = 1,   // is in the training set if game.set & 1 is true
      validation = 2, // is in validation set
      batch = 3,      // is in active minibatch
      test = 4        // is in test set
    } set;
  };

  // data on one player
  struct Player {
    std::string name;
    int lastOccurrence; // max index of game where this player participated or -1
  };

  // load the games listed in the path, optionally with move features from featuresDir.
  void load(const std::string& path, const std::string& featureDir = "");
  void store(const std::string& path) const;
  // retrieve up to bufsize moves played by the player in games before the game index, return # retrieved
  size_t getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize);
  // randomly assign the `set` member of every game; *Part in [0.0, 1.0], testPart = 1-trainingPart-validationPart
  void randomSplit(Rand& rand, float trainingPart, float validationPart);
  // randomly assign set=batch to the given nr of training games (and reset previous batch to set=training)
  void randomBatch(Rand& rand, size_t batchSize);
  // set the batch marker on all games that contain up to windowSize recent moves for any marked game
  void markRecentGames(int windowSize, Logger* logger = nullptr);
  // read features of the full model (kata trunk output)
  std::vector<float> readFeaturesFromFile(const std::string& featurePath);
  // read features of the proof of concept model (6 features / move)
  std::vector<MoveFeatures> readPocFeaturesFromFile(const std::string& featurePath);

  std::vector<Game> games;
  std::vector<Player> players;

  static const uint32_t FEATURE_HEADER; // magic bytes for feature file
  static const uint32_t FEATURE_HEADER_POC; // magic bytes for feature file (proof of concept model)

private:

  std::map<std::string, std::size_t> nameIndex;  // player names to unique index into player_

  std::size_t getOrInsertNameIndex(const std::string& name);  // insert with lastOccurrence
  void loadPocFeatures(const std::string& featureDir);

};

#endif // STRMODEL_DATASET_H
