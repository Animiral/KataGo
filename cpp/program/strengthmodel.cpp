#include "strengthmodel.h"
#include "core/global.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

using std::vector;
using std::cout;
using std::cerr;

// poor man's pre-C++20 format, https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string custom_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

StrengthModel::StrengthModel(Search& search_) noexcept
: search(&search_)
{
}

void StrengthModel::getMoveFeatures(const char* sgfPath, vector<MoveFeatures>& blackFeatures, vector<MoveFeatures>& whiteFeatures) const
{
    auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
    if(NULL == sgf)
        throw IOError(string("Failed to open SGF: ") + sgfPath + ".");
    cout << "Evaluate game \"" << sgf->fileName << "\": " << sgf->rootNode.getSingleProperty("PB") << " vs " << sgf->rootNode.getSingleProperty("PW") << "\n";

    cout << "Got " << sgf->moves.size() << " moves in SGF.\n";
    Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
    Board board;
    BoardHistory history;
    Player pla;
    sgf->setupInitialBoardAndHist(rules, board, pla, history);
    NNResultBuf nnResultBuf;

    for(Move move : sgf->moves) {
      const auto locStr = Location::toString(move.loc, sgf->xSize, sgf->ySize);
      cout << "Player " << PlayerIO::playerToString(move.pla) << " moves at " << locStr << ". ";

      // === get raw NN eval and features ===
      bool skipCache = false;
      bool isRoot = true;
      SearchParams searchParams = search->searchParams;
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
      nnInputParams.conservativePassAndIsRoot = searchParams.conservativePass && isRoot;
      nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
      nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
      nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == move.pla;
      nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
      if(searchParams.playoutDoublingAdvantage != 0) {
        Player playoutDoublingAdvantagePla = searchParams.playoutDoublingAdvantagePla == C_EMPTY ? move.pla : searchParams.playoutDoublingAdvantagePla;
        nnInputParams.playoutDoublingAdvantage = (
          getOpp(pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
        );
      }
      if(searchParams.ignorePreRootHistory || searchParams.ignoreAllHistory)
        nnInputParams.maxHistory = 0;
      search->nnEvaluator->evaluate(board, history, move.pla, nnInputParams, nnResultBuf, skipCache, false);
      assert(nnResultBuf.hasResult);
      const NNOutput& nnout = *nnResultBuf.result;
      cout << custom_format("NN output: win=%.2f, loss=%.2f, scoreMean=%.2f, lead=%.2f\n", nnout.whiteWinProb, nnout.whiteLossProb, nnout.whiteScoreMean, nnout.whiteLead);
      float movePolicy = nnout.policyProbs[nnout.getPos(move.loc, board)];
      float maxPolicy = *std::max_element(std::begin(nnout.policyProbs), std::end(nnout.policyProbs));
      cout << custom_format("  policy at %s=%.2f, maxPolicy=%.2f\n", locStr.c_str(), movePolicy, maxPolicy);

      if(P_WHITE == pla) {
          MoveFeatures mf{nnout.whiteWinProb, nnout.whiteLead, movePolicy, maxPolicy};
          whiteFeatures.push_back(mf);
      }
      else {
          MoveFeatures mf{nnout.whiteLossProb, -nnout.whiteLead, movePolicy, maxPolicy};
          blackFeatures.push_back(mf);
      }
      // === search ===

      // search.setPosition(pla, board, history);
      // search.runWholeSearch(move.pla, false);
      // // use search results
      // Loc chosenLoc = search.getChosenMoveLoc();
      // const SearchNode* node = search.rootNode;
      // double sharpScore;
      // suc = search.getSharpScore(node, sharpScore);
      // cout << "SharpScore: " << sharpScore << "\n";
      // vector<AnalysisData> data;
      // search.getAnalysisData(data, 1, false, 50, false);
      // cout << "Got " << data.size() << " analysis data. \n";

      // ReportedSearchValues values = search.getRootValuesRequireSuccess();

      // cout << "Root values:\n "
      //      << "  win:" << values.winValue << "\n"
      //      << "  loss:" << values.lossValue << "\n"
      //      << "  noResult:" << values.noResultValue << "\n"
      //      << "  staticScoreValue:" << values.staticScoreValue << "\n"
      //      << "  dynamicScoreValue:" << values.dynamicScoreValue << "\n"
      //      << "  expectedScore:" << values.expectedScore << "\n"
      //      << "  expectedScoreStdev:" << values.expectedScoreStdev << "\n"
      //      << "  lead:" << values.lead << "\n"
      //      << "  winLossValue:" << values.winLossValue << "\n"
      //      << "  utility:" << values.utility << "\n"
      //      << "  weight:" << values.weight << "\n"
      //      << "  visits:" << values.visits << "\n";

      // apply move
      bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
      if(!suc)
        cerr << "Illegal move " << PlayerIO::playerToString(pla) << " at " << Location::toString(move.loc, sgf->xSize, sgf->ySize) << "\n";
    }
}

float StrengthModel::rating(const vector<MoveFeatures>& features) const
{
    float winProbSum = 0;
    for(auto mf : features)
        winProbSum += mf.winProb;
    return winProbSum / features.size();
}

RatingSystem::RatingSystem(StrengthModel& model) noexcept
: strengthModel(&model)
{
}

void RatingSystem::calculate(string sgfList, string outFile)
{
	map< string, vector<MoveFeatures> > playerHistory;
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

    std::ofstream ostrm(outFile);
    if (!ostrm.is_open())
        throw IOError("Could not write SGF list to " + outFile);

    ostrm << "File,Player White,Player Black,Winner,WhiteWinrate,BlackRating,WhiteRating\n"; // header

    while (std::getline(istrm, line)) {
        std::istringstream iss(line);
        std::string sgfPath; std::getline(iss, sgfPath, ',');
        std::string whiteName; std::getline(iss, whiteName, ',');
        std::string blackName; std::getline(iss, blackName, ',');
        std::string winner; std::getline(iss, winner, ','), std::getline(iss, winner); // skip one field for real winner
        std::cout << blackName << " vs " << whiteName << ": " << winner << "\n";

        // determine winner and count
        float blackRating = playerRating[blackName] = strengthModel->rating(playerHistory[blackName]);
        float whiteRating = playerRating[whiteName] = strengthModel->rating(playerHistory[whiteName]);
        // // get winner by higher rating
        if(winner[0] == 'B' && blackRating > whiteRating) successCount++;
        if(winner[0] == 'W' && whiteRating > blackRating) successCount++;
        sgfCount++;

        // expand player histories with new move features
        strengthModel->getMoveFeatures(sgfPath.c_str(), playerHistory[blackName], playerHistory[whiteName]);

        // file output
        size_t bufsize = sgfPath.size() + whiteName.size() + blackName.size() + winner.size() + 100;
        std::unique_ptr<char[]> buffer( new char[ bufsize ] );
        int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%s,%.2f,%.2f,%.2f\n",
            sgfPath.c_str(), whiteName.c_str(), blackName.c_str(), winner.c_str(), .5f + (whiteRating - blackRating)/2, blackRating, whiteRating);
        if(printed <= 0)
            throw IOError( "Error during formatting." );
        ostrm << buffer.get();
    }

    ostrm.close();
    istrm.close();

    successRate = float(successCount) / sgfCount;
    successLogp = logp;
}
