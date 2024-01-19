#ifndef PROGRAM_STRENGTHMODEL_H_
#define PROGRAM_STRENGTHMODEL_H_

#include <string>
#include <vector>
#include <map>
#include "neuralnet/strengthnet.h"
#include "core/rand.h"
#include "search/search.h"
#include "dataio/sgf.h"


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
    float score;             // game outcome for black: 0 for loss, 1 for win
    Prediction prediction;
    
    enum {
      training = 0,   // is in the training set if ~game.set & 1 is true
      validation = 1, // is in validation set
      batch = 2,      // is in active minibatch
      test = 3        // is in test set
    } set;
  };

  // data on one player
  struct Player {
    std::string name;
    int lastOccurrence; // max index of game where this player participated or -1
  };

  void load(const std::string& path, const std::string& featureDir);
  void store(const std::string& path) const;
  // retrieve up to bufsize moves played by the player in games before the game index, return # retrieved
  size_t getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize);
  // randomly assign the `set` member of every game; *Part in [0.0, 1.0], testPart = 1-trainingPart-validationPart
  void randomSplit(Rand& rand, float trainingPart, float validationPart);
  // randomly assign set=batch to the given nr of training games (and reset previous batch to set=training)
  void randomBatch(Rand& rand, size_t batchSize);

  std::vector<Game> games;
  std::vector<Player> players;

  static const uint32_t FEATURE_HEADER; // magic bytes for feature file

private:

  std::map<std::string, std::size_t> nameIndex;  // player names to unique index into player_

  std::size_t getOrInsertNameIndex(const std::string& name);  // insert with lastOccurrence
  std::vector<MoveFeatures> readFeaturesFromFile(const std::string& featurePath);

};

// The predictor, given a match between two opponents, estimates their ratings and the match score (win probability).
// This is the abstract base class for our predictors:
//   - The StochasticPredictor based on simple statistics
//   - The SmallPredictor based on the StrengthNet
//   - The FullPredictor (to be done!)
class Predictor {

public:

  // The resulting prediction might keep the players' ratings at 0 (no prediction), but it always predicts the score.
  virtual Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) = 0;

protected:

  // give an expected score by assuming that the given ratings are Elo ratings.
  static float eloScore(float blackRating, float whiteRating);

};

class StochasticPredictor : public Predictor {

public:

  Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) override;

};

class SmallPredictor : public Predictor {

public:

  explicit SmallPredictor(StrengthNet& strengthNet) noexcept; // ownership of the net remains with the caller
  Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) override;

private:

  StrengthNet* net;

};

using FeaturesAndTargets = std::vector<std::pair<StrengthNet::Input, StrengthNet::Output> >;

// This class encapsulates the main strength model functions on a dataset.
// It can extract features (=one-time preprocessing with Kata net), train our model, and evaluate it.
class StrengthModel
{

public:

  // cache calculated move features for every sgfPath under featureDir
  explicit StrengthModel(const std::string& strengthModelFile_, const std::string& featureDir_) noexcept;

  FeaturesAndTargets getFeaturesAndTargets(const Dataset& dataset) const;
  // Analyze SGF with KataGo network and search to determine the embedded features of every move
  static void extractGameFeatures(const CompactSgf& sgf, const Search& search, std::vector<MoveFeatures>& blackFeatures, std::vector<MoveFeatures>& whiteFeatures);

  // training loop, save result to file
  void train(FeaturesAndTargets& xy, size_t split, int epochs, size_t batchSize, float weightPenalty, float learnrate);

  // run predictions using windowSize moves and determine rate/error over the games matching the given set (set=batch games also match set=training)
  struct Evaluation {
    float sqerr;  // sum of squares of predicted ratings vs player ratings
    float rate;   // relative amount of matches in which the predicted winner matched the actual winner
    float logp;   // cumulative log-likelihood of all match outcomes in the eyes of the predictor
  };
  Evaluation evaluate(Dataset& dataset, Predictor& predictor, int set, size_t windowSize = 1000);

  std::string featureDir;
  StrengthNet net;

private:

  std::string strengthModelFile;
  // Dataset dataset; // TODO: refactor to have the same dataset for feature extraction, training and evaluation

  bool maybeWriteMoveFeaturesCached(const std::string& cachePath, const std::vector<MoveFeatures>& features) const;

};

#endif  // PROGRAM_STRENGTHMODEL_H_
