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

bool GameFeatures::present() const noexcept {
  return !blackFeatures.empty() && !whiteFeatures.empty();
}

StrengthModel::StrengthModel(const string& strengthModelFile, Search& search_, const string& featureDir_) noexcept
  : search(&search_), featureDir(featureDir_)
{}

namespace {
string cachePath(const string& featureDir, const string& sgfPath, Player player) {
  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  return custom_format("%s/%s_%sFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str(), PlayerIO::playerToString(player).c_str());
}
}

GameFeatures StrengthModel::getGameFeatures(const string& sgfPath) const {
  string blackFeaturesPath, whiteFeaturesPath;
  GameFeatures features = maybeGetGameFeaturesCachedForSgf(sgfPath, blackFeaturesPath, whiteFeaturesPath);
  if(features.present())
    return features;

  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  if(NULL == sgf)
    throw IOError(string("Failed to open SGF: ") + sgfPath + ".");
  cout << "Evaluate game \"" << sgf->fileName << "\": " << sgf->rootNode.getSingleProperty("PB") << " vs " << sgf->rootNode.getSingleProperty("PW") << "\n";

  return extractGameFeatures(*sgf, blackFeaturesPath, whiteFeaturesPath);
}

GameFeatures StrengthModel::getGameFeatures(const CompactSgf& sgf) const {
  string blackFeaturesPath, whiteFeaturesPath;
  GameFeatures features = maybeGetGameFeaturesCachedForSgf(sgf.fileName, blackFeaturesPath, whiteFeaturesPath);
  if(features.present())
    return features;

  return extractGameFeatures(sgf, blackFeaturesPath, whiteFeaturesPath);
}

GameFeatures StrengthModel::maybeGetGameFeaturesCachedForSgf(const string& sgfPath, string& blackFeaturesPath, string& whiteFeaturesPath) const {
  GameFeatures features;
  if(featureDir.empty()) {
    blackFeaturesPath = whiteFeaturesPath = "";
    return features;
  }

  blackFeaturesPath = cachePath(featureDir, sgfPath, P_BLACK);
  whiteFeaturesPath = cachePath(featureDir, sgfPath, P_WHITE);
  features.blackFeatures = maybeGetMoveFeaturesCached(blackFeaturesPath);
  features.whiteFeatures = maybeGetMoveFeaturesCached(whiteFeaturesPath);
  if(features.blackFeatures.empty() ^ features.whiteFeatures.empty())
    cerr << "Incomplete feature cache for " << sgfPath << "\n";
  return features;
}

vector<MoveFeatures> StrengthModel::maybeGetMoveFeaturesCached(const string& cachePath) const {
  vector<MoveFeatures> features;
  auto cacheFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(cachePath.c_str(), "rb"), &std::fclose);
  if(nullptr == cacheFile)
    return features;
  uint32_t header; // must match
  size_t readcount = std::fread(&header, 4, 1, cacheFile.get());
  if(1 != readcount || FEATURE_HEADER != header)
    return features;
  while(!std::feof(cacheFile.get())) {
    MoveFeatures mf;
    readcount = std::fread(&mf, sizeof(MoveFeatures), 1, cacheFile.get());
    if(1 == readcount)
      features.push_back(mf);
  }
  return features;
}

bool StrengthModel::maybeWriteMoveFeaturesCached(const string& cachePath, const vector<MoveFeatures>& features) const {
  if(featureDir.empty() || features.empty())
    return false;
  string cacheDir = FileUtils::dirname(cachePath);
  if(!FileUtils::create_directories(cacheDir))
    return false;
  auto cacheFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(cachePath.c_str(), "wb"), &std::fclose);
  if(nullptr == cacheFile)
    return false;
  size_t writecount = std::fwrite(&FEATURE_HEADER, 4, 1, cacheFile.get());
  if(1 != writecount)
    return false;
  writecount = std::fwrite(features.data(), sizeof(MoveFeatures), features.size(), cacheFile.get());
  if(features.size() != writecount)
    return false;
  if(0 != std::fclose(cacheFile.release()))
    return false;
  return true;
}

GameFeatures StrengthModel::extractGameFeatures(const CompactSgf& sgf, const string& blackFeaturesPath, const string& whiteFeaturesPath) const {
  GameFeatures features;
  const auto& moves = sgf.moves;
  Rules rules = sgf.getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf.setupInitialBoardAndHist(rules, board, initialPla, history);

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
      cerr << "Illegal move " << PlayerIO::playerToString(move.pla) << " at " << Location::toString(move.loc, sgf.xSize, sgf.ySize) << "\n";
      return {};
    }

    // === get raw NN eval and features ===
    search->nnEvaluator->evaluate(board, history, getOpp(move.pla), nnInputParams, nnResultBuf, false, false);
    assert(nnResultBuf.hasResult);
    nnout = nnResultBuf.result.get();

    if(P_WHITE == move.pla) {
      features.whiteFeatures.push_back({
        nnout->whiteWinProb, nnout->whiteLead, movePolicy, maxPolicy,
        prevWhiteWinProb-nnout->whiteWinProb, prevWhiteLead-nnout->whiteLead
      });
    }
    else {
      features.blackFeatures.push_back({
        nnout->whiteLossProb, -nnout->whiteLead, movePolicy, maxPolicy,
        prevWhiteLossProb-nnout->whiteLossProb, -prevWhiteLead+nnout->whiteLead
      });
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
  }

  // save to cache
  bool blackCached = maybeWriteMoveFeaturesCached(blackFeaturesPath, features.blackFeatures);
  bool whiteCached = maybeWriteMoveFeaturesCached(whiteFeaturesPath, features.whiteFeatures);
  if(!blackCached || !whiteCached)
    cerr << "Failed to save cached features for " << sgf.fileName << ".\n";

  return features;
}

namespace {

float fSum(float a[], size_t N) noexcept {  // sum with slightly better numerical stability
  if(N <= 0)
    return 0;
  for(size_t step = 1; step < N; step *= 2) {
    for(size_t i = 0; i+step < N; i += 2*step)
      a[i] += a[i+step];
  }
  return a[0];
}

float fAvg(float a[], size_t N) noexcept {
  return fSum(a, N) / N;
}

float fVar(float a[], size_t N, float avg) noexcept { // corrected variance
  for(size_t i = 0; i < N; i++)
    a[i] = (a[i]-avg)*(a[i]-avg);
  return fSum(a, N) / (N-1);
}

float normcdf(float x) noexcept {
  return .5f * (1.f + std::erf(x / std::sqrt(2.f)));
}

void copyPloss(float a[], const vector<MoveFeatures>& m, size_t N) noexcept {
  for(size_t i = 0; i < N; i++)
    a[i] = m[i + m.size() - N].pointsLoss;
}

}

float StrengthModel::rating(const vector<MoveFeatures>& history) const {
  vector<float> ploss(history.size());
  for(size_t i = 0; i < history.size(); i++)
    ploss[i] = history[i].pointsLoss;
  return 20.f - fAvg(ploss.data(), ploss.size());
}

float StrengthModel::whiteWinrate(const vector<MoveFeatures>& whiteHistory, const vector<MoveFeatures>& blackHistory) const {
  constexpr float gamelength = 100; // assume 100 moves per player for an average game
  constexpr size_t window = 1000; // only sample the most recent moves
  const size_t wN = std::min(whiteHistory.size(), window);
  const size_t bN = std::min(blackHistory.size(), window);
  vector<float> buffer(window);
  copyPloss(buffer.data(), whiteHistory, wN);
  float wplavg = fAvg(vector<float>(buffer).data(), wN) * gamelength;  // average white points loss
  float wplvar = fVar(buffer.data(), wN, wplavg) * gamelength;  // variance of white points loss
  copyPloss(buffer.data(), blackHistory, bN);
  float bplavg = fAvg(vector<float>(buffer).data(), bN) * gamelength;  // average black points loss
  float bplvar = fVar(buffer.data(), bN, bplavg) * gamelength;  // variance of black points loss
  float wstdadv = (bplavg - wplavg) / std::sqrt(bplvar + wplvar); // white pt advantage in standard normal distribution at move# [2*gamelength]
  return normcdf(wstdadv);
}

const uint32_t StrengthModel::FEATURE_HEADER = 0xfea70235;

RatingSystem::RatingSystem(StrengthModel& model) noexcept
: strengthModel(&model) {
}

void RatingSystem::calculate(const string& sgfList, const string& featureDir, const string& outFile) {
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
    winner = Global::toLower(Global::trim(winner));
    std::cout << blackName << " vs " << whiteName << ": " << winner << "\n";

    // determine winner and count
    float blackRating = playerRating[blackName] = strengthModel->rating(playerHistory[blackName]);
    float whiteRating = playerRating[whiteName] = strengthModel->rating(playerHistory[whiteName]);
    float whiteWinrate = strengthModel->whiteWinrate(playerHistory[whiteName], playerHistory[blackName]);
    // // get winner by higher rating
    float z = .5f; // result representation
    if(Global::isPrefix(winner,"b+") || Global::isPrefix(winner,"black+")) z = 0.f;
    if(Global::isPrefix(winner,"w+") || Global::isPrefix(winner,"white+")) z = 1.f;
    float winnerPred = std::abs(1.f - z - whiteWinrate);
    if(winnerPred > .5f)
      successCount++;
    logp += std::log(winnerPred);
    sgfCount++;

    // expand player histories with new move features
    GameFeatures features = strengthModel->getGameFeatures(sgfPath);
    playerHistory[blackName].insert(playerHistory[blackName].end(), features.blackFeatures.begin(), features.blackFeatures.end());
    playerHistory[whiteName].insert(playerHistory[whiteName].end(), features.whiteFeatures.begin(), features.whiteFeatures.end());

    // file output
    size_t bufsize = sgfPath.size() + whiteName.size() + blackName.size() + winner.size() + 100;
    std::unique_ptr<char[]> buffer( new char[ bufsize ] );
    int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%s,%.2f,%.2f,%.2f\n",
      sgfPath.c_str(), whiteName.c_str(), blackName.c_str(), winner.c_str(), whiteWinrate, blackRating, whiteRating);
    if(printed <= 0)
      throw IOError( "Error during formatting." );
    ostrm << buffer.get();
  }

  ostrm.close();
  istrm.close();

  successRate = float(successCount) / sgfCount;
  successLogp = logp;
}
