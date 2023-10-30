#include "strengthmodel.h"
#include "core/global.h"
#include <iostream>
#include <fstream>
#include <sstream>

using std::vector;

RatingSystem::RatingSystem(StrengthModel& model) noexcept
: strengthModel(&model)
{
}

void RatingSystem::calculate(string sgfList, string outFile)
{
	map< string, vector<string> > playerHistory;
	int successCount = 0;
	int sgfCount = 0;
	float logp = 0;

    std::ifstream istrm(sgfList);
    if (!istrm.is_open())
    	throw IOError("Could not read SGF list from " + sgfList);

    std::string line;
    std::getline(istrm, line); // ignore first line
    if(!istrm)
    	throw IOError("Could not read header line from " + sgfList);

    while (std::getline(istrm, line)) {
        std::istringstream iss(line);
        std::string sgfPath; std::getline(iss, sgfPath, ',');
        std::string whiteName; std::getline(iss, whiteName, ',');
        std::string blackName; std::getline(iss, blackName, ',');
        std::string winner; std::getline(iss, winner, ','), std::getline(iss, winner); // skip one field for real winner
        std::cout << blackName << " vs " << whiteName << ": " << winner << "\n";

        // float blackRating = playerRating[blackName] = strengthModel->rating(playerHistory[blackName]);
        // float whiteRating = playerRating[whiteName] = strengthModel->rating(playerHistory[whiteName]);
        // // get winner by higher rating
        // if(winner[0] == 'B' && blackRating > whiteRating) successCount++;
        // if(winner[0] == 'W' && whiteRating > blackRating) successCount++;
        sgfCount++;

        // expand player histories
        playerHistory[blackName].push_back(sgfPath);
        playerHistory[whiteName].push_back(sgfPath);
    }

    successRate = float(successCount) / sgfCount;
    successLogp = logp;
}
