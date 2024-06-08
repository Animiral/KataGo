#ifndef STRMODEL_DATASET_H
#define STRMODEL_DATASET_H

#include <memory>
#include <ostream>
#include "core/logger.h"
#include "core/rand.h"
#include "game/board.h"
#include "neuralnet/strengthnet.h"
#include "strmodel/using.h"

namespace StrModel {

using GameId = int; // index into dataset.games
using PlayerId = int; // index into dataset.players

// Identifies specific boards by turn index in multiple games.
// Turns are maintained in ascending order.
struct GamesTurns {
  map<GameId, vector<int>> bygame;
};

using FeatureVector = vector<float>;

// Holds strength model features on the board position at move index and maybe the players' move in that position.
struct BoardFeatures {
  int turn; // 0-based move number in the game
  Player pla; // only from precomputation, not loaded or stored in feature files
  int pos; // index into trunk data of move chosen by player or -1
  shared_ptr<FeatureVector> trunk; // trunk output data or nullptr
  shared_ptr<FeatureVector> pick; // pick output data or nullptr
  shared_ptr<FeatureVector> head; // head features or nullptr
};

// represents a set of moves, possibly spread over several games
// // TODO: separate into Query (which recent move data to acquire from NN) and Results
// struct SelectedMoves {
//   struct Move {
//     int index; // 0-based move number in the game
//     Selection selection; // which features to query for this move
//     // output values: filled later
//     Player pla;
//     shared_ptr<FeatureVector> trunk; // trunk output data
//     shared_ptr<FeatureVector> pick; // pick output data
//     shared_ptr<FeatureVector> head; // head features
//     int pos; // index into trunk data of move chosen by player
//   };
//   struct Moveset {
//     std::vector<Move> moves; // in ascending order
//     Player pla; // player associated with the moveset (even if it contains different-color moves)
//     void insert(int index, Selection selection); // prevents duplicates
//     void merge(const Moveset& rhs); // merge rhs entries into this
//     bool hasAllResults() const; // true if all player moves have data
//     void releaseStorage(); // free memory by letting go of TrunkOutputs/PickOutputs (can be re-read from zip later)
//     std::pair<Moveset, Moveset> splitBlackWhite() const;
//     void writeToZip(const std::string& filePath) const;
//     void printSummary(std::ostream& stream) const; // list contained moves in readable format for debugging
//     static Moveset readFromZip(const std::string& filePath, Player pla = 0);

//    private:
//     constexpr static int nnXLen = 19;
//     constexpr static int nnYLen = 19;
//     constexpr static int numTrunkFeatures = 384;  // strength model is limited to this size
//     constexpr static int trunkSize = nnXLen*nnYLen*numTrunkFeatures;
//     constexpr static int numHeadFeatures = 6;      // filling struct MoveFeatures
//   };

//   std::map<std::string, Moveset> bygame;

//   size_t size() const; // count selected moves
//   void merge(const SelectedMoves& rhs); // merge rhs entries into this
//   void copyFeaturesFrom(const SelectedMoves& rhs); // get trunk/pos data into this
// };

// Load and store dataset files: SGFs and feature ZIPs.
// This class is virtual and can be abstracted away, e.g. mocked for testing purposes.
class DatasetFiles {

public:

  explicit DatasetFiles(const string& featureDir = "");
  DatasetFiles(const DatasetFiles& ) = delete; // no slicing
  virtual ~DatasetFiles() noexcept;

  // Determine the path to the feature file for the given game, player and title ("Features" or "Recent")
  string featurePath(const string& sgfPath, Player pla, const char* title) const;
  virtual vector<BoardFeatures> loadFeatures(const string& path) const;
  virtual void storeFeatures(const vector<BoardFeatures>& features, const string& path) const;

private:

  string featureDir;

};

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
      PlayerId player; // index of player (given as name string in CSV file)
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

  // load the games listed in the CSV stream
  Dataset(std::istream& stream, const DatasetFiles& files);
  Dataset(const string& path, const DatasetFiles& files); // convenience c'tor
  void load(std::istream& stream);
  void store(const std::string& path) const;
  ::Player playerColor(PlayerId playerId, GameId gameId) const;
  // identify up to capacity moves played by the player in games before the game index (without attached data)
  // with head features, also returns follow-up (opponent) moves and final board state
  GamesTurns getRecentMoves(::Player player, GameId game, size_t capacity) const;
  // randomly assign the `set` member of every game; *Part in [0.0, 1.0], testPart = 1-trainingPart-validationPart
  void randomSplit(Rand& rand, float trainingPart, float validationPart);
  // randomly assign set=batch to the given nr of training games (and reset previous batch to set=training)
  void randomBatch(Rand& rand, size_t batchSize);

  vector<BoardFeatures> loadFeatures(GameId gameId, ::Player pla, const char* title) const;
  void storeFeatures(const vector<BoardFeatures>& features, GameId gameId, ::Player pla, const char* title) const;

  vector<Game> games;
  vector<Player> players;

private:

  const DatasetFiles* files;
  map<string, size_t> nameIndex;  // player names to unique index into players

  size_t getOrInsertNameIndex(const string& name);  // insert with lastOccurrence

};

} // end namespace StrModel

#endif // STRMODEL_DATASET_H
