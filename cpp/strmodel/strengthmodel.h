#ifndef PROGRAM_STRENGTHMODEL_H_
#define PROGRAM_STRENGTHMODEL_H_

#include "neuralnet/strengthnet.h"
#include "core/rand.h"
#include "search/search.h"
#include "dataio/sgf.h"
#include "strmodel/dataset.h"

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

  // give an expected score by assuming that the given ratings are Glicko-2 ratings.
  static float glickoScore(float blackRating, float whiteRating);

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

// This class encapsulates the main strength model functions on a dataset.
// It can extract features (=one-time preprocessing with Kata net), train our model, and evaluate it.
class StrengthModel
{

public:

  // load the strength model or random-initialize a new one
  explicit StrengthModel(const std::string& strengthModelFile = "", Dataset* dataset_ = nullptr, Rand* rand = nullptr) noexcept;

  // use KataGo network to precompute features for every dataset game and write them to feature files
  void extractFeatures(const std::string& featureDir, const Search& search, Logger* logger = nullptr);

  // training loop on strength network
  void train(int epochs, int steps, size_t batchSize, float weightPenalty, float learnrate, size_t windowSize, Rand& rand);

  // run predictions using windowSize moves and determine rate/error over the games matching the given set,
  // set=batch games also match set=training, set==-1 matches everything
  struct Evaluation {
    size_t count; // how many games were evaluated in the selected set
    float mse;   // mean of squares of predicted ratings vs player ratings
    float rate;  // relative amount of matches in which the predicted winner matched the actual winner
    float logp;  // mean log-likelihood of all match outcomes in the eyes of the predictor
  };
  Evaluation evaluate(Predictor& predictor, int set, size_t windowSize = 1000);

  // run the strength model on SGFs provided directly by the user and predict one rating number from them
  struct Analysis {
    size_t moveCount;
    float avgWRLoss;
    float avgPLoss;
    float rating;
  };
  Analysis analyze(std::vector<Sgf*> sgfs, const std::string& playerName, const Search& search);

  StrengthNet net;

private:

  Dataset* dataset;

  // Analyze SGF with KataGo network and search to determine the embedded features of every move
  static void extractGameFeatures(const CompactSgf& sgf, const Search& search, std::vector<MoveFeatures>& blackFeatures, std::vector<MoveFeatures>& whiteFeatures);
  void writeFeaturesToFile(const std::string& featurePath, const std::vector<MoveFeatures>& features) const;

};

#endif  // PROGRAM_STRENGTHMODEL_H_
