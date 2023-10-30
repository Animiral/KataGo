#include <string>
#include <map>
#include <filesystem>

using std::string;
using std::map;

// The strength model uses an additional trained neural network to derive rating from
// given player history.
class StrengthModel
{

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

private:

	StrengthModel* strengthModel;
	map<string, float> playerRating;
	float successRate;
	float successLogp;

};
