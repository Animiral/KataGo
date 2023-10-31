#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include "search/search.h"
#include "dataio/sgf.h"

using std::string;
using std::vector;
using std::map;

// this is what we give as input to the strength model for a single move
struct MoveFeatures
{
  float winProb;
  float lead;
  float movePolicy;
  float maxPolicy;
  float winrateLoss; // compared to previous move
  float pointsLoss; // compared to previous move
};

// The strength model uses an additional trained neural network to derive rating from
// given player history.
class StrengthModel
{

public:

	explicit StrengthModel(Search& search_) noexcept;

	// Analyze SGF and use the strength model to determine the embedded features of every move
	void getMoveFeatures(const char* sgfPath, vector<MoveFeatures>& blackFeatures, vector<MoveFeatures>& whiteFeatures) const;

	// Predict rating of player given the moves from their games
	float rating(const vector<MoveFeatures>& features) const;

private:

	Search* search;

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
	void calculate(string sgfList, string outFile);

	map<string, float> playerRating;
	float successRate;
	float successLogp;

private:

	StrengthModel* strengthModel;

};
