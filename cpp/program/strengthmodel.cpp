#include "strengthmodel.h"
#include "core/global.h"
#include "core/fileutils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <memory>
#include "core/using.h"

using std::map;

namespace {

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

}

void Dataset::load(const string& path, const std::string& featureDir) {
  std::ifstream istrm(path);
  if (!istrm.is_open())
    throw IOError("Could not read dataset from " + path);

  std::string line;
  std::getline(istrm, line);
  if(!istrm)
    throw IOError("Could not read header line from " + path);
  line = Global::trim(line);

  // map known fieldnames to row indexes, wherever they may be
  enum class F { ignore, sgfPath, whiteName, blackName, whiteLabel, blackLabel, winner };
  vector<F> fields;
  std::string field;
  std::istringstream iss(line);
  while(std::getline(iss, field, ',')) {
    if("File" == field) fields.push_back(F::sgfPath);
    else if("Player White" == field) fields.push_back(F::whiteName);
    else if("Player Black" == field) fields.push_back(F::blackName);
    else if("WhiteLabel" == field) fields.push_back(F::whiteLabel);
    else if("BlackLabel" == field) fields.push_back(F::blackLabel);
    else if("Winner" == field || "Judgement" == field) fields.push_back(F::winner);
    else fields.push_back(F::ignore);
  }

  while (std::getline(istrm, line)) {
    size_t gameIndex = games.size();
    games.emplace_back();
    Game& game = games[gameIndex];

    line = Global::trim(line);
    iss = std::istringstream(line);
    int fieldIndex = 0;
    while(std::getline(iss, field, ',')) {
      switch(fields[fieldIndex++]) {
      case F::sgfPath:
        game.sgfPath = field;
        break;
      case F::whiteName:
        game.whitePlayer = getOrInsertNameIndex(field);
        break;
      case F::blackName:
        game.blackPlayer = getOrInsertNameIndex(field);
        break;
      case F::whiteLabel:
        game.whiteRating = Global::stringToFloat(field);
        break;
      case F::blackLabel:
        game.blackRating = Global::stringToFloat(field);
        break;
      case F::winner:
        if('b' == field[0] || 'B' == field[0])
          game.score = 1;
        else if('w' == field[0] || 'W' == field[0])
          game.score = 0;
        break;
      default:
      case F::ignore:
        break;
      }
    }
    if(!istrm)
      throw IOError("Error while reading from " + path);
    game.prevWhiteGame = players[game.whitePlayer].lastOccurrence;
    game.prevBlackGame = players[game.blackPlayer].lastOccurrence;

    string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
    string blackFeaturesPath = Global::strprintf("%s/%s_BlackFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    game.blackFeatures = readFeaturesFromFile(blackFeaturesPath);
    game.whiteFeatures = readFeaturesFromFile(whiteFeaturesPath);

    players[game.whitePlayer].lastOccurrence = gameIndex;
    players[game.blackPlayer].lastOccurrence = gameIndex;
  }

  istrm.close();
}

void Dataset::store(const string& path) const {
  std::ofstream ostrm(path);
  if (!ostrm.is_open())
    throw IOError("Could not write SGF list to " + path);

  ostrm << "File,Player White,Player Black,Score,BlackRating,WhiteRating,PredictedScore,PredictedBlackRating,PredictedWhiteRating\n"; // header

  for(const Game& game : games) {
    string blackName = players[game.blackPlayer].name;
    string whiteName = players[game.whitePlayer].name;

    // file output
    size_t bufsize = game.sgfPath.size() + whiteName.size() + blackName.size() + 100;
    std::unique_ptr<char[]> buffer( new char[ bufsize ] );
    int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%.2f,%.2f,%.2f,%f,%f,%f\n",
      game.sgfPath.c_str(), whiteName.c_str(), blackName.c_str(),
      game.score, game.blackRating, game.whiteRating,
      game.predictedScore, game.predictedBlackRating, game.predictedWhiteRating);
    if(printed <= 0)
      throw IOError("Error during formatting.");
    ostrm << buffer.get();
  }

  ostrm.close();
}

const uint32_t Dataset::FEATURE_HEADER = 0xfea70235;

size_t Dataset::getOrInsertNameIndex(const std::string& name) {
  auto it = nameIndex.find(name);
  if(nameIndex.end() == it) {
    size_t index = players.size();
    players.push_back({name, 0});
    bool success;
    std::tie(it, success) = nameIndex.insert({name, index});
  }
  return it->second;
}

vector<MoveFeatures> Dataset::readFeaturesFromFile(const string& featurePath) {
  vector<MoveFeatures> features;
  auto featureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(featurePath.c_str(), "rb"), &std::fclose);
  if(nullptr == featureFile)
    return features;
  uint32_t header; // must match
  size_t readcount = std::fread(&header, 4, 1, featureFile.get());
  if(1 != readcount || FEATURE_HEADER != header)
    return features;
  while(!std::feof(featureFile.get())) {
    MoveFeatures mf;
    readcount = std::fread(&mf, sizeof(MoveFeatures), 1, featureFile.get());
    if(1 == readcount)
      features.push_back(mf);
  }
  return features;
}

StrengthModel::StrengthModel(const string& strengthModelFile_, Search* search_, const string& featureDir_) noexcept
  : featureDir(featureDir_), strengthModelFile(strengthModelFile_), net(), search(search_)
{
  if(!net.loadModelFile(strengthModelFile)) {
    cerr << "Could not load existing strength model from " << strengthModelFile << ". Random-initializing new strength model.\n";
    Rand rand; // TODO: allow seeding from outside StrengthModel
    net.randomInit(rand);
  }
}

StrengthModel::StrengthModel(const string& strengthModelFile_, Search& search_, const string& featureDir_) noexcept
  : StrengthModel(strengthModelFile_, &search_, featureDir_)
{}

FeaturesAndTargets StrengthModel::getFeaturesAndTargets(const Dataset& dataset) const {
  FeaturesAndTargets featuresTargets;
  for(const Dataset::Game& gm : dataset.games) {
    featuresTargets.emplace_back(gm.blackFeatures, gm.blackRating);
    featuresTargets.emplace_back(gm.whiteFeatures, gm.whiteRating);
  }
  return featuresTargets;
}

void StrengthModel::extractGameFeatures(const CompactSgf& sgf, const Search& search, vector<MoveFeatures>& blackFeatures, vector<MoveFeatures>& whiteFeatures) {
  const auto& moves = sgf.moves;
  Rules rules = sgf.getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf.setupInitialBoardAndHist(rules, board, initialPla, history);

  NNResultBuf nnResultBuf;
  MiscNNInputParams nnInputParams;
  // evaluate initial board once for initial prev-features
  search.nnEvaluator->evaluate(board, history, initialPla, nnInputParams, nnResultBuf, false, false);
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
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf.xSize, sgf.ySize)));

    // === get raw NN eval and features ===
    search.nnEvaluator->evaluate(board, history, getOpp(move.pla), nnInputParams, nnResultBuf, false, false);
    assert(nnResultBuf.hasResult);
    nnout = nnResultBuf.result.get();

    if(P_WHITE == move.pla) {
      whiteFeatures.push_back({
        nnout->whiteWinProb, nnout->whiteLead, movePolicy, maxPolicy,
        prevWhiteWinProb-nnout->whiteWinProb, prevWhiteLead-nnout->whiteLead
      });
    }
    else {
      blackFeatures.push_back({
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
}

void StrengthModel::train(FeaturesAndTargets& xy, size_t split, int epochs, size_t batchSize, float weightPenalty, float learnrate) {
  assert(split <= xy.size());
  Rand rand; // TODO: allow seeding from outside StrengthModel
  net.randomInit(rand);
  batchSize = 1; // TODO: properly implement batches

  for(int e = 0; e < epochs; e++) {
    float grads_var = 0;
    std::shuffle(&xy[0], &xy[split], rand);
    // train weights
    for(int i = 0; i < split; i += batchSize) {
      net.setBatchSize(std::min(batchSize, split-i));
      for(size_t b = 0; i+b < split && b < batchSize; b++) {
        net.setInput(xy[i+b].first);
        net.forward();
        // cout << "Sample #" << i << "(" << xy[i].first.size() << " moves): (" << y_hat << "-" << xy[i].second << ")^2 = " << (y_hat-xy[i].second)*(y_hat-xy[i].second) << "\n";
        net.backward(xy[i+b].second/*, b*/);
        grads_var += net.gradsVar();
      }
      net.mergeGrads();

      // if(e % 5 == 4 && i == 0) {
      //   net.printWeights(cout, "epoch " + Global::intToString(e));
      //   net.printState(cout, "epoch " + Global::intToString(e));
      //   // cout << "Test #" << i-split << " (" << xy[i].first.size() << " moves): prediction=" << std::fixed << std::setprecision(3) << y_hat << ", target=" << xy[i].second << ", sqerr=" << sqerr << "\n";
      // }

      net.update(weightPenalty, learnrate);
    }
    grads_var /= split; // average in 1 training update
    // net.printWeights(cout, "epoch " + Global::intToString(e));
    // net.printState(cout, "epoch " + Global::intToString(e));

    // test epoch result
    float mse_training = 0; // error on training set
    for(int i = 0; i < split; i++) {
      net.setInput(xy[i].first);
      net.forward();
      float y_hat = net.getOutput();
      float sqerr = (y_hat - xy[i].second) * (y_hat - xy[i].second);
      mse_training += sqerr;
    }
    mse_training /= split;

    float mse = 0;
    for(int i = split; i < xy.size(); i++) {
      net.setInput(xy[i].first);
      net.forward();
      float y_hat = net.getOutput();
      float sqerr = (y_hat - xy[i].second) * (y_hat - xy[i].second);
      mse += sqerr;
    }
    mse /= xy.size() - split;
    float theta_var = net.thetaVar();
    // cout << "Epoch " << e << ": mse=" << std::fixed << std::setprecision(3) << mse << "\n";
    cout << custom_format("Epoch %d: mse_training=%.2f, mse=%.2f, theta^2=%.4f, grad^2=%.4f\n", e, mse_training, mse, theta_var, grads_var);
  }
  net.saveModelFile(strengthModelFile);
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
  size_t writecount = std::fwrite(&Dataset::FEATURE_HEADER, 4, 1, cacheFile.get());
  if(1 != writecount)
    return false;
  writecount = std::fwrite(features.data(), sizeof(MoveFeatures), features.size(), cacheFile.get());
  if(features.size() != writecount)
    return false;
  if(0 != std::fclose(cacheFile.release()))
    return false;
  return true;
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

RatingSystem::RatingSystem(StrengthModel& model) noexcept
: strengthModel(&model) {
}

void RatingSystem::calculate(const string& sgfList, const string& outFile) {
  Dataset dataset;
  dataset.load(sgfList, strengthModel->featureDir);
  map< size_t, vector<MoveFeatures> > playerHistory;
  int successCount = 0;
  int sgfCount = 0;
  float logp = 0;

  for(Dataset::Game& gm : dataset.games) {
    string blackName = dataset.players[gm.blackPlayer].name;
    string whiteName = dataset.players[gm.whitePlayer].name;
    string winner = gm.score > .5 ? "B+":"W+";
    std::cout << blackName << " vs " << whiteName << ": " << winner << "\n";

    // determine winner and count
    gm.predictedBlackRating = playerRating[gm.blackPlayer] = strengthModel->rating(playerHistory[gm.blackPlayer]);
    gm.predictedWhiteRating = playerRating[gm.whitePlayer] = strengthModel->rating(playerHistory[gm.whitePlayer]);
    float whiteWinrate = strengthModel->whiteWinrate(playerHistory[gm.whitePlayer], playerHistory[gm.blackPlayer]);
    gm.predictedScore = 1 - whiteWinrate;
    float winnerPred = std::abs(gm.score - whiteWinrate);
    if(winnerPred > .5f)
      successCount++;
    logp += std::log(winnerPred);
    sgfCount++;

    // expand player histories with new move features
    playerHistory[gm.blackPlayer].insert(playerHistory[gm.blackPlayer].end(), gm.blackFeatures.begin(), gm.blackFeatures.end());
    playerHistory[gm.whitePlayer].insert(playerHistory[gm.whitePlayer].end(), gm.whiteFeatures.begin(), gm.whiteFeatures.end());
  }

  dataset.store(outFile);

  successRate = float(successCount) / sgfCount;
  successLogp = logp;
}
