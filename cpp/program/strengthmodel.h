#ifndef PROGRAM_STRENGTHMODEL_H_
#define PROGRAM_STRENGTHMODEL_H_

#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include "neuralnet/strengthnet.h"
#include "search/search.h"
#include "dataio/sgf.h"

using std::string;
using std::vector;
using std::pair;
using std::map;

// data on one game from the dataset list file
struct GameMeta {
  std::string sgfPath;
  std::string whiteName;
  std::string blackName;
  std::string whiteLabel;
  std::string blackLabel;
  std::string winner;
};

using Dataset = vector<GameMeta>;

// all the strength model relevant information extracted from a game
struct GameFeatures {
  vector<MoveFeatures> blackFeatures;
  vector<MoveFeatures> whiteFeatures;

  bool present() const noexcept;
};

using FeaturesAndTargets = vector<pair<StrengthNet::Input, StrengthNet::Output> >;

// The strength model uses an additional trained neural network to derive rating from
// given player history.
class StrengthModel
{

public:

  // cache calculated move features for every sgfPath under featureDir
  explicit StrengthModel(const string& strengthModelFile_, Search* search_, const string& featureDir_) noexcept;
  explicit StrengthModel(const string& strengthModelFile_, Search& search_, const string& featureDir_) noexcept;

  // Analyze SGF and use the strength model to determine the embedded features of every move
  GameFeatures getGameFeatures(const string& sgfPath) const;
  GameFeatures getGameFeatures(const CompactSgf& sgf) const;
  // get dataset from a list file
  static Dataset loadDataset(const string& path);
  FeaturesAndTargets getFeaturesAndTargets(const Dataset& dataset) const;

  // training loop, save result to file
  void train(const FeaturesAndTargets& xy, size_t split, int epochs, float learnrate);

  // Predict rating of player given the moves from their games
  float rating(const vector<MoveFeatures>& history) const;

  // Predict winning chances given the moves from their games
  float whiteWinrate(const vector<MoveFeatures>& whiteHistory, const vector<MoveFeatures>& blackHistory) const;

private:

  string strengthModelFile;
  StrengthNet net;
  Search* search;
  string featureDir;
  static const uint32_t FEATURE_HEADER;

  GameFeatures maybeGetGameFeaturesCachedForSgf(const string& sgfPath, string& blackFeaturesPath, string& whiteFeaturesPath) const;
  vector<MoveFeatures> maybeGetMoveFeaturesCached(const string& cachePath) const;
  bool maybeWriteMoveFeaturesCached(const string& cachePath, const vector<MoveFeatures>& features) const;
  GameFeatures extractGameFeatures(const CompactSgf& sgf, const string& blackFeaturesPath, const string& whiteFeaturesPath) const;

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
  void calculate(const string& sgfList, const string& outFile);

  map<string, float> playerRating;
  float successRate;
  float successLogp;

private:

  StrengthModel* strengthModel;

};

#endif  // PROGRAM_STRENGTHMODEL_H_
