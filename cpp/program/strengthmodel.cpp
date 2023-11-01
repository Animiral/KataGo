#include "strengthmodel.h"
#include "core/global.h"
#include "core/fileutils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <memory>

using std::vector;
using std::cout;
using std::cerr;

// poor man's pre-C++20 format, https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string custom_format( const std::string& format, Args ... args ) {
  int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
  if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
  auto size = static_cast<size_t>( size_s );
  std::unique_ptr<char[]> buf( new char[ size ] );
  std::snprintf( buf.get(), size, format.c_str(), args ... );
  return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

StrengthModel::StrengthModel(Search& search_, const char* featureDir_) noexcept
: search(&search_), featureDir(featureDir_) {
  if(nullptr != featureDir && std::strlen(featureDir) <= 0)
    featureDir = nullptr;
}

void StrengthModel::getMoveFeatures(const char* sgfPath, vector<MoveFeatures>& blackFeatures, vector<MoveFeatures>& whiteFeatures) const {
  string blackFeaturesPath, whiteFeaturesPath; // cache files
  size_t blackBegin = blackFeatures.size();
  size_t blackEnd = blackBegin;
  size_t whiteBegin = whiteFeatures.size();
  size_t whiteEnd = whiteBegin;
  bool gotCachedFeatures = false;
  if(nullptr != featureDir) {
    size_t len = std::strlen(sgfPath);
    assert(0 == std::strcmp(".sgf", sgfPath + len - 4));
    string sgfPathWithoutExt(sgfPath, len - 4);
    blackFeaturesPath = custom_format( "%s/%s_blackFeatures.bin", featureDir, sgfPathWithoutExt.c_str());
    whiteFeaturesPath = custom_format( "%s/%s_whiteFeatures.bin", featureDir, sgfPathWithoutExt.c_str());
    bool gotBlackCache = maybeGetMoveFeaturesCached(blackFeaturesPath.c_str(), blackFeatures);
    bool gotWhiteCache = maybeGetMoveFeaturesCached(whiteFeaturesPath.c_str(), whiteFeatures);
    gotCachedFeatures = gotBlackCache && gotWhiteCache;
    if(gotBlackCache ^ gotWhiteCache)
      cerr << "Incomplete feature cache for " << sgfPath << "\n";
  }

  if(!gotCachedFeatures)
    getMoveFeaturesNoCache(sgfPath, blackFeatures, whiteFeatures);

  blackEnd = blackFeatures.size();
  whiteEnd = whiteFeatures.size();
  float bwloss=0.f, bploss=0.f, wwloss=0.f, wploss=0.f;
  for(size_t i = blackBegin; i < blackEnd; i++) {
    bwloss += blackFeatures[i].winrateLoss;
    bploss += blackFeatures[i].pointsLoss;
  }
  for(size_t i = whiteBegin; i < whiteEnd; i++) {
    wwloss += whiteFeatures[i].winrateLoss;
    wploss += whiteFeatures[i].pointsLoss;
  }
  cout << custom_format("Avg win%% loss: w %.3f, b %.3f; pt loss: w %.3f, b %.3f\n",
    wwloss/(whiteEnd-whiteBegin), bwloss/(blackEnd-blackBegin), wploss/(whiteEnd-whiteBegin), bploss/(blackEnd-blackBegin));

  if(nullptr != featureDir && !gotCachedFeatures) {
    bool blackCached = maybeWriteMoveFeaturesCached(blackFeaturesPath.c_str(), blackFeatures.data()+blackBegin, blackFeatures.data()+blackEnd);
    bool whiteCached = maybeWriteMoveFeaturesCached(whiteFeaturesPath.c_str(), whiteFeatures.data()+whiteBegin, whiteFeatures.data()+whiteEnd);
    if(!blackCached || !whiteCached)
      cerr << "Failed to save cached features for " << sgfPath << ".\n";
  }
}

bool StrengthModel::maybeGetMoveFeaturesCached(const char* cachePath, vector<MoveFeatures>& features) const {
  auto cacheFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(cachePath, "rb"), &std::fclose);
  if(nullptr == cacheFile)
    return false;
  uint32_t header; // must match
  size_t readcount = std::fread(&header, 4, 1, cacheFile.get());
  if(1 != readcount || FEATURE_HEADER != header)
    return false;
  while(!std::feof(cacheFile.get())) {
    MoveFeatures mf;
    readcount = std::fread(&mf, sizeof(MoveFeatures), 1, cacheFile.get());
    if(1 == readcount)
      features.push_back(mf);
  }
  return true;
}

bool StrengthModel::maybeWriteMoveFeaturesCached(const char* cachePath, const MoveFeatures* begin, const MoveFeatures* end) const {
  if(nullptr == begin || nullptr == end)
    return false;
  string cacheDir = FileUtils::dirname(cachePath);
  if(!FileUtils::create_directories(cacheDir.c_str()))
    return false;
  auto cacheFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(cachePath, "wb"), &std::fclose);
  if(nullptr == cacheFile)
    return false;
  size_t writecount = std::fwrite(&FEATURE_HEADER, 4, 1, cacheFile.get());
  if(1 != writecount)
    return false;
  writecount = std::fwrite(begin, sizeof(MoveFeatures), end-begin, cacheFile.get());
  if(end-begin != writecount)
    return false;
  if(0 != std::fclose(cacheFile.release()))
    return false;
  return true;
}

void StrengthModel::getMoveFeaturesNoCache(const char* sgfPath, vector<MoveFeatures>& blackFeatures, vector<MoveFeatures>& whiteFeatures) const {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  if(NULL == sgf)
    throw IOError(string("Failed to open SGF: ") + sgfPath + ".");
  cout << "Evaluate game \"" << sgf->fileName << "\": " << sgf->rootNode.getSingleProperty("PB") << " vs " << sgf->rootNode.getSingleProperty("PW") << "\n";

  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  NNResultBuf nnResultBuf;
  MiscNNInputParams nnInputParams;
  // evaluate initial board once for initial prev-features
  search->nnEvaluator->evaluate(board, history, initialPla, nnInputParams, nnResultBuf, false, false);
  assert(nnResultBuf.hasResult);
  const NNOutput* nnout = nnResultBuf.result.get();
  float prevWhiteWinProb = nnout->whiteWinProb;
  float prevWhiteLossProb = nnout->whiteLossProb;
  float prevWhiteLead = nnout->whiteLead;
  float movePolicy = moves.empty() ? 0.f : nnout->policyProbs[nnout->getPos(moves[0].loc, board)];
  float maxPolicy = *std::max_element(std::begin(nnout->policyProbs), std::end(nnout->policyProbs));

  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc) {
      cerr << "Illegal move " << PlayerIO::playerToString(move.pla) << " at " << Location::toString(move.loc, sgf->xSize, sgf->ySize) << "\n";
      return;
    }

    // === get raw NN eval and features ===
    search->nnEvaluator->evaluate(board, history, getOpp(move.pla), nnInputParams, nnResultBuf, false, false);
    assert(nnResultBuf.hasResult);
    nnout = nnResultBuf.result.get();

    MoveFeatures mf;
    if(P_WHITE == move.pla) {
      mf = MoveFeatures{nnout->whiteWinProb, nnout->whiteLead, movePolicy, maxPolicy,
                        prevWhiteWinProb-nnout->whiteWinProb, prevWhiteLead-nnout->whiteLead};
      whiteFeatures.push_back(mf);
    }
    else {
      mf = MoveFeatures{nnout->whiteLossProb, -nnout->whiteLead, movePolicy, maxPolicy,
                        prevWhiteLossProb-nnout->whiteLossProb, -prevWhiteLead+nnout->whiteLead};
      blackFeatures.push_back(mf);
    }

    // === search ===

    // SearchParams searchParams = search->searchParams;
    // nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
    // nnInputParams.conservativePassAndIsRoot = searchParams.conservativePass;
    // nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
    // nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
    // nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == move.pla;
    // nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
    // if(searchParams.playoutDoublingAdvantage != 0) {
    //   Player playoutDoublingAdvantagePla = searchParams.playoutDoublingAdvantagePla == C_EMPTY ? move.pla : searchParams.playoutDoublingAdvantagePla;
    //   nnInputParams.playoutDoublingAdvantage = (
    //     getOpp(pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    //   );
    // }
    // if(searchParams.ignorePreRootHistory || searchParams.ignoreAllHistory)
    //   nnInputParams.maxHistory = 0;

    // search.setPosition(move.pla, board, history);
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

    prevWhiteWinProb = nnout->whiteWinProb;
    prevWhiteLossProb = nnout->whiteLossProb;
    prevWhiteLead = nnout->whiteLead;
    movePolicy = turnIdx >= moves.size()-1 ? 0.f : nnout->policyProbs[nnout->getPos(moves[turnIdx+1].loc, board)]; // policy of next move
    maxPolicy = *std::max_element(std::begin(nnout->policyProbs), std::end(nnout->policyProbs));

    cout << custom_format("Player %s moves at %s. Features: ploss=%.2f, vloss=%.2f, lead=%.2f.\n",
      PlayerIO::playerToString(move.pla).c_str(), Location::toString(move.loc, sgf->xSize, sgf->ySize).c_str(),
      mf.winrateLoss, mf.pointsLoss, mf.lead);
  }
}

float StrengthModel::rating(const vector<MoveFeatures>& features) const {
  float winProbSum = 0;
  for(auto mf : features)
    winProbSum += mf.winProb;
  return winProbSum / features.size();
}

const uint32_t StrengthModel::FEATURE_HEADER = 0xfea70235;

RatingSystem::RatingSystem(StrengthModel& model) noexcept
: strengthModel(&model) {
}

void RatingSystem::calculate(string sgfList, string featureDir, string outFile) {
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
