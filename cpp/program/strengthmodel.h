#ifndef PROGRAM_STRENGTHMODEL_H_
#define PROGRAM_STRENGTHMODEL_H_

#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include "neuralnet/strengthnet.h"
#include "search/search.h"
#include "dataio/sgf.h"


// The dataset is a chronological sequence of games with move features.
class Dataset {

public:

  // data on one game from the dataset list file
  struct Game {
    std::string sgfPath;
    std::size_t whitePlayer; // index of white player (given as name string in list file)
    std::size_t blackPlayer; // index of black player (given as name string in list file)
    float whiteRating;       // rating number target
    float blackRating;       // rating number target
    float score;             // game outcome for black: 0 for loss, 1 for win
    float predictedWhiteRating; // prediction by model
    float predictedBlackRating; // prediction by model
    float predictedScore;       // prediction by model
    std::size_t prevWhiteGame; // index of most recent game with white player before this
    std::size_t prevBlackGame; // index of most recent game with black player before this
    std::vector<MoveFeatures> blackFeatures;
    std::vector<MoveFeatures> whiteFeatures;
  };

  // data on one player
  struct Player {
    std::string name;
    size_t lastOccurrence; // max index of game where this player participated
  };

  void load(const std::string& path, const std::string& featureDir);
  void store(const std::string& path) const;

  std::vector<Game> games;
  std::vector<Player> players;

  static const uint32_t FEATURE_HEADER; // magic bytes for feature file

private:

  std::map<std::string, std::size_t> nameIndex;  // player names to unique index into player_

  std::size_t getOrInsertNameIndex(const std::string& name);  // insert with lastOccurrence
  std::vector<MoveFeatures> readFeaturesFromFile(const std::string& featurePath);

};

using FeaturesAndTargets = std::vector<std::pair<StrengthNet::Input, StrengthNet::Output> >;

// The strength model uses an additional trained neural network to derive rating from
// given player history.
class StrengthModel
{

public:

  // cache calculated move features for every sgfPath under featureDir
  explicit StrengthModel(const std::string& strengthModelFile_, Search* search_, const std::string& featureDir_) noexcept;
  explicit StrengthModel(const std::string& strengthModelFile_, Search& search_, const std::string& featureDir_) noexcept;

  FeaturesAndTargets getFeaturesAndTargets(const Dataset& dataset) const;
  // Analyze SGF with KataGo network and search to determine the embedded features of every move
  static void extractGameFeatures(const CompactSgf& sgf, const Search& search, std::vector<MoveFeatures>& blackFeatures, std::vector<MoveFeatures>& whiteFeatures);

  // training loop, save result to file
  void train(FeaturesAndTargets& xy, size_t split, int epochs, size_t batchSize, float weightPenalty, float learnrate);

  // Predict rating of player given the moves from their games
  float rating(const std::vector<MoveFeatures>& history) const;

  // Predict winning chances given the moves from their games
  float whiteWinrate(const std::vector<MoveFeatures>& whiteHistory, const std::vector<MoveFeatures>& blackHistory) const;

  std::string featureDir;

private:

  std::string strengthModelFile;
  StrengthNet net;
  Search* search;

  bool maybeWriteMoveFeaturesCached(const std::string& cachePath, const std::vector<MoveFeatures>& features) const;

};

// The RatingSystem can process a set of game records from a common rating pool.
// For each game, we predicts the likely winner using the StrengthModel on the
// recent prior game history of both players.
// We write all predictions to an output file and tally up two quality measurements:
// the rate of accurate predictions and the accumulated log-likelihood of accurate predictions.
class RatingSystem
{

public:

  explicit RatingSystem(StrengthModel& model) noexcept;
  // process the SGF list file, store extracted features under featureDir, write processed list to outFile
  void calculate(const std::string& sgfList, const std::string& outFile);

  std::map<std::size_t, float> playerRating;
  float successRate;
  float successLogp;

private:

  StrengthModel* strengthModel;

};

#endif  // PROGRAM_STRENGTHMODEL_H_
