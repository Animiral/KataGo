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
  enum class F { ignore, sgfPath, whiteName, blackName, whiteLabel, blackLabel, winner, set };
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
    else if("Set" == field) fields.push_back(F::set);
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
        game.white.player = getOrInsertNameIndex(field);
        break;
      case F::blackName:
        game.black.player = getOrInsertNameIndex(field);
        break;
      case F::whiteLabel:
        game.white.rating = Global::stringToFloat(field);
        break;
      case F::blackLabel:
        game.black.rating = Global::stringToFloat(field);
        break;
      case F::winner:
        if('b' == field[0] || 'B' == field[0])
          game.score = 1;
        else if('w' == field[0] || 'W' == field[0])
          game.score = 0;
        break;
      case F::set:
        if("t" == field || "T" == field) game.set = Game::training;
        if("v" == field || "V" == field) game.set = Game::validation;
        if("e" == field || "E" == field) game.set = Game::test;
        break;
      default:
      case F::ignore:
        break;
      }
    }
    if(!istrm)
      throw IOError("Error while reading from " + path);
    game.white.prevGame = players[game.white.player].lastOccurrence;
    game.black.prevGame = players[game.black.player].lastOccurrence;

    players[game.white.player].lastOccurrence = gameIndex;
    players[game.black.player].lastOccurrence = gameIndex;
  }

  istrm.close();

  if(!featureDir.empty())
    loadFeatures(featureDir);
}

void Dataset::store(const string& path) const {
  std::ofstream ostrm(path);
  if (!ostrm.is_open())
    throw IOError("Could not write SGF list to " + path);

  ostrm << "File,Player White,Player Black,Score,BlackRating,WhiteRating,PredictedScore,PredictedBlackRating,PredictedWhiteRating,Set\n"; // header

  for(const Game& game : games) {
    string blackName = players[game.black.player].name;
    string whiteName = players[game.white.player].name;

    // file output
    size_t bufsize = game.sgfPath.size() + whiteName.size() + blackName.size() + 100;
    std::unique_ptr<char[]> buffer( new char[ bufsize ] );
    int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%f,%.2f,%.2f,%f,%f,%f,%c\n",
      game.sgfPath.c_str(), whiteName.c_str(), blackName.c_str(),
      game.score, game.black.rating, game.white.rating,
      game.prediction.score, game.prediction.blackRating, game.prediction.whiteRating, "TVBE"[game.set]);
    if(printed <= 0)
      throw IOError("Error during formatting.");
    ostrm << buffer.get();
  }

  ostrm.close();
}

size_t Dataset::getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize) {
  assert(player < players.size());
  assert(game <= games.size());

  // start from the game preceding the specified index
  int gameIndex;
  if(games.size() == game) {
      gameIndex = players[player].lastOccurrence;
  }
  else {
    Game* gm = &games[game];
    if(player == gm->black.player)
      gameIndex = gm->black.prevGame;
    else if(player == gm->white.player)
      gameIndex = gm->white.prevGame;
    else
      gameIndex = static_cast<int>(game) - 1;
  }

  // go backwards in player's history and fill the buffer in backwards order
  MoveFeatures* outptr = buffer + bufsize;
  while(gameIndex >= 0 && outptr > buffer) {
    while(gameIndex >= 0 && player != games[gameIndex].black.player && player != games[gameIndex].white.player)
      gameIndex--; // this is just defense to ensure that we find a game which the player occurs in
    if(gameIndex < 0)
      break;
    Game* gm = &games[gameIndex];
    bool isBlack = player == gm->black.player;
    const auto& features = isBlack ? gm->black.features : gm->white.features;
    for(int i = features.size(); i > 0 && outptr > buffer;)
      *--outptr = features[--i];
    gameIndex = isBlack ? gm->black.prevGame : gm->white.prevGame;
  }

  // if there are not enough features in history to fill the buffer, adjust
  size_t count = bufsize - (outptr - buffer);
  if(outptr > buffer)
    std::memmove(buffer, outptr, count * sizeof(MoveFeatures));
  return count;
}

void Dataset::randomSplit(Rand& rand, float trainingPart, float validationPart) {
  assert(trainingPart >= 0);
  assert(validationPart >= 0);
  assert(trainingPart + validationPart <= 1);
  size_t N = games.size();
  vector<uint32_t> gameIdxs(N);
  rand.fillShuffledUIntRange(N, gameIdxs.data());
  size_t trainingCount = std::llround(trainingPart * N);
  size_t validationCount = std::llround(validationPart * N);
  for(size_t i = 0; i < trainingCount; i++)
    games[gameIdxs[i]].set = Game::training;
  for(size_t i = trainingCount; i < trainingCount + validationCount && i < N; i++)
    games[gameIdxs[i]].set = Game::validation;
  for(size_t i = trainingCount + validationCount; i < N; i++)
    games[gameIdxs[i]].set = Game::test;
}

void Dataset::randomBatch(Rand& rand, size_t batchSize) {
  vector<size_t> trainingIdxs;
  for(size_t i = 0; i < games.size(); i++)
    if(~games[i].set & 1)
      trainingIdxs.push_back(i);
  batchSize = std::min(batchSize, trainingIdxs.size());
  vector<uint32_t> batchIdxs(trainingIdxs.size());
  rand.fillShuffledUIntRange(trainingIdxs.size(), batchIdxs.data());
  for(size_t i = 0; i < batchSize; i++)
    games[trainingIdxs[batchIdxs[i]]].set = Game::batch;
  for(size_t i = batchSize; i < batchIdxs.size(); i++)
    games[trainingIdxs[batchIdxs[i]]].set = Game::training;
}

const uint32_t Dataset::FEATURE_HEADER = 0xfea70235;

size_t Dataset::getOrInsertNameIndex(const std::string& name) {
  auto it = nameIndex.find(name);
  if(nameIndex.end() == it) {
    size_t index = players.size();
    players.push_back({name, -1});
    bool success;
    std::tie(it, success) = nameIndex.insert({name, index});
  }
  return it->second;
}

void Dataset::loadFeatures(const std::string& featureDir) {
  for(Game& game : games) {
    string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
    string blackFeaturesPath = Global::strprintf("%s/%s_BlackFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    game.black.features = readFeaturesFromFile(blackFeaturesPath);
    game.white.features = readFeaturesFromFile(whiteFeaturesPath);
  }
}

vector<MoveFeatures> Dataset::readFeaturesFromFile(const string& featurePath) {
  vector<MoveFeatures> features;
  auto featureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(featurePath.c_str(), "rb"), &std::fclose);
  if(nullptr == featureFile)
    throw IOError("Failed to read access feature file " + featurePath);
  uint32_t header; // must match
  size_t readcount = std::fread(&header, 4, 1, featureFile.get());
  if(1 != readcount || FEATURE_HEADER != header)
    throw IOError("Failed to read from feature file " + featurePath);
  while(!std::feof(featureFile.get())) {
    MoveFeatures mf;
    readcount = std::fread(&mf, sizeof(MoveFeatures), 1, featureFile.get());
    if(1 == readcount)
      features.push_back(mf);
  }
  return features;
}


float Predictor::eloScore(float blackRating, float whiteRating) {
  float Qblack = static_cast<float>(std::pow(10, blackRating / 400));
  float Qwhite = static_cast<float>(std::pow(10, whiteRating / 400));
  return Qblack / (Qblack + Qwhite);
}

// TODO: adapt code from OGS/goratings into glickoScore
    // def expected_win_probability(self, white: "Glicko2Entry", handicap_adjustment: float, ignore_g: bool = False) -> float:
    //     # Implementation extracted from glicko2_update below.
    //     if not ignore_g:
    //         def g() -> float:
    //             return 1
    //     else:
    //         def g() -> float:
    //             return 1 / sqrt(1 + (3 * white.phi ** 2) / (pi ** 2))

    //     E = 1 / (1 + exp(-g() * (self.rating + handicap_adjustment - white.rating) / GLICKO2_SCALE))
    //     return E

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

}

Dataset::Prediction StochasticPredictor::predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) {
  if(0 == blackCount || 0 == whiteCount)
    return {0, 0, .5}; // no data for prediction
  constexpr float gamelength = 100; // assume 100 moves per player for an average game
  vector<float> buffer(std::max(blackCount, whiteCount));
  for(size_t i = 0; i < whiteCount; i++)
    buffer[i] = whiteFeatures[i].pointsLoss;
  float wplavg = fAvg(vector<float>(buffer).data(), whiteCount);  // average white points loss
  float wplvar = 2 <= whiteCount ? fVar(buffer.data(), whiteCount, wplavg) : 100.f;      // variance of white points loss
  for(size_t i = 0; i < blackCount; i++)
    buffer[i] = blackFeatures[i].pointsLoss;
  float bplavg = fAvg(vector<float>(buffer).data(), blackCount);  // average black points loss
  float bplvar = 2 <= blackCount ? fVar(buffer.data(), blackCount, bplavg) : 100.f;      // variance of black points loss
  const float epsilon = 0.000001f;  // avoid div by 0
  float z = std::sqrt(gamelength) * (wplavg - bplavg) / std::sqrt(bplvar + wplvar + epsilon); // white pt advantage in standard normal distribution at move# [2*gamelength]
  return {0, 0, normcdf(z)};
}

SmallPredictor::SmallPredictor(StrengthNet& strengthNet) noexcept : net(&strengthNet)
{}

Dataset::Prediction SmallPredictor::predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) {
  Dataset::Prediction prediction;
  net->setInput({vector<MoveFeatures>(blackFeatures, blackFeatures + blackCount)});
  net->forward();
  prediction.blackRating = net->getOutput()[0];
  net->setInput({vector<MoveFeatures>(whiteFeatures, whiteFeatures + whiteCount)});
  net->forward();
  prediction.whiteRating = net->getOutput()[0];
  prediction.score = eloScore(prediction.blackRating, prediction.whiteRating);
  return prediction;
}


StrengthModel::StrengthModel(const string& strengthModelFile, Dataset* dataset_) noexcept
  : net(), dataset(dataset_)
{
  bool loaded = false;
  if(!strengthModelFile.empty() && !(loaded = net.loadModelFile(strengthModelFile))) {
    cerr << "Could not load existing strength model from " << strengthModelFile << ". Random-initializing new strength model.\n";
  }
  if(!loaded) {
    Rand rand;
    net.randomInit(rand);
  }
}

void StrengthModel::extractFeatures(const std::string& featureDir, const Search& search) {
  for(Dataset::Game& game : dataset->games) {
    auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(game.sgfPath));
    extractGameFeatures(*sgf, search, game.black.features, game.white.features);
    string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
    string blackFeaturesPath = Global::strprintf("%s/%s_BlackFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    if(FileUtils::exists(blackFeaturesPath))
      writeFeaturesToFile(blackFeaturesPath, game.black.features);
    if(FileUtils::exists(whiteFeaturesPath))
      writeFeaturesToFile(whiteFeaturesPath, game.white.features);
  }
}

void StrengthModel::train(int epochs, int steps, size_t batchSize, float weightPenalty, float learnrate, size_t windowSize) {
  Rand rand; // TODO: allow seeding from outside StrengthModel
  SmallPredictor predictor(net);

  for(int e = 0; e < epochs; e++) {
    float grads_var = 0;
    // train weights
    for(int s = 0; s < steps; s++) {
      dataset->randomBatch(rand, batchSize);
      vector<vector<MoveFeatures>> inputs;
      vector<float> targets;
      for(size_t t = 0; t < dataset->games.size(); t++) {
        const Dataset::Game& game = dataset->games[t];
        if(Dataset::Game::batch == game.set) {
          for(auto& playerInfo : {game.white, game.black}) {
            vector<MoveFeatures> features(windowSize);
            size_t moveCount = dataset->getRecentMoves(playerInfo.player, t, features.data(), windowSize);
            features.resize(moveCount);
            inputs.push_back(features);
            targets.push_back(playerInfo.rating);
          }
        }
      }
      net.setInput(inputs);
      net.forward();
      // if(0 == s) {
      //   auto outt = net.getOutput();
      //   outt.resize(10);
      //   cout << "Forward output: ";
      //   for(auto o : outt)
      //     cout << o << ", ";
      //   cout << "\n";
      //   cout << "Backward targets: ";
      //   auto tt = targets;
      //   tt.resize(10);
      //   for(auto o : tt)
      //     cout << o << ", ";
      //   cout << "\n";
      // }
      net.setTarget(targets);
      net.backward();
      grads_var += net.gradsVar();
      // if(s == 0) {
      //   net.printWeights(cout, "epoch " + Global::intToString(e) + " step " + Global::intToString(s));
      //   net.printState(cout, "epoch " + Global::intToString(e) + " step " + Global::intToString(s));
      //   net.printGrads(cout, "epoch " + Global::intToString(e) + " step " + Global::intToString(s));
      //   // cout << "Test #" << i-split << " (" << xy[i].first.size() << " moves): prediction=" << std::fixed << std::setprecision(3) << y_hat << ", target=" << xy[i].second << ", sqerr=" << sqerr << "\n";
      // }

      net.update(weightPenalty, learnrate);
    }
    // cout << "Sample #" << i << "(" << xy[i].first.size() << " moves): (" << y_hat << "-" << xy[i].second << ")^2 = " << (y_hat-xy[i].second)*(y_hat-xy[i].second) << "\n";
    grads_var /= steps; // average in 1 training update
    // net.printWeights(cout, "epoch " + Global::intToString(e));
    // net.printState(cout, "epoch " + Global::intToString(e));

    // test epoch result
    Evaluation trainingEval = evaluate(predictor, Dataset::Game::training, windowSize);
    Evaluation validationEval = evaluate(predictor, Dataset::Game::validation, windowSize);
    float theta_var = net.thetaVar();
    // cout << "Epoch " << e << ": mse=" << std::fixed << std::setprecision(3) << mse << "\n";
    cout << Global::strprintf("Epoch %d: mse_training=%.2f, mse_validation=%.2f, theta^2=%.4f, grad^2=%.4f\n", e, trainingEval.mse, validationEval.mse, theta_var, grads_var);
  }
}

StrengthModel::Evaluation StrengthModel::evaluate(Predictor& predictor, int set, size_t windowSize) {
  vector<MoveFeatures> blackFeatures(windowSize);
  vector<MoveFeatures> whiteFeatures(windowSize);
  int successCount = 0;
  int sgfCount = 0;
  float sqerr = 0;
  float logp = 0;

  for(size_t i = 0; i < dataset->games.size(); i++) {
    Dataset::Game& gm = dataset->games[i];

    // always perform prediction, even for games not in the set
    size_t blackCount = dataset->getRecentMoves(gm.black.player, i, blackFeatures.data(), windowSize);
    size_t whiteCount = dataset->getRecentMoves(gm.white.player, i, whiteFeatures.data(), windowSize);
    gm.prediction = predictor.predict(blackFeatures.data(), blackCount, whiteFeatures.data(), whiteCount);

    if(gm.set != set && !(Dataset::Game::training == set && Dataset::Game::batch == gm.set))
      continue;

    float diffBlack = gm.black.rating - gm.prediction.blackRating;
    float diffWhite = gm.white.rating - gm.prediction.whiteRating;
    sqerr += diffBlack * diffBlack + diffWhite * diffWhite;
    float winnerPred = 1 - std::abs(gm.score - gm.prediction.score);
    if(winnerPred > .5f)
      successCount++;
    logp += std::log(winnerPred);
    sgfCount++;
  }

  float rate = float(successCount) / sgfCount;
  float mse = sqerr / sgfCount;
  return { mse, rate, logp };
}

StrengthModel::Analysis StrengthModel::analyze(vector<Sgf*> sgfs, const string& playerName, const Search& search) {
  vector<MoveFeatures> playerFeatures;
  for (const auto* sgf : sgfs)
  {
    Player p;
    if(sgf->getPlayerName(P_BLACK) == playerName)
      p = P_BLACK;
    else if(sgf->getPlayerName(P_WHITE) == playerName)
      p = P_WHITE;
    else {
      cerr << "Player \"" << playerName << "\" not found in " << sgf->fileName << ".\n";
      continue;
    }
    vector<MoveFeatures> blackFeatures, whiteFeatures;
    extractGameFeatures(CompactSgf(sgf), search, blackFeatures, whiteFeatures);
    if(P_BLACK == p)
      playerFeatures.insert(playerFeatures.end(), blackFeatures.begin(), blackFeatures.end());
    if(P_WHITE == p)
      playerFeatures.insert(playerFeatures.end(), whiteFeatures.begin(), whiteFeatures.end());
  }

  Analysis analysis;
  for(const auto& mf : playerFeatures) {
    analysis.avgWRLoss += mf.winrateLoss;
    analysis.avgPLoss += mf.pointsLoss;
  }
  size_t N = playerFeatures.size();
  analysis.avgWRLoss /= N;
  analysis.avgPLoss /= N;
  net.setInput({playerFeatures});
  net.forward();
  vector<float> rating = net.getOutput();
  analysis.rating = rating[0];
  return analysis;
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

void StrengthModel::writeFeaturesToFile(const string& featurePath, const vector<MoveFeatures>& features) const {
  string featureDir = FileUtils::dirname(featurePath);
  if(!FileUtils::create_directories(featureDir))
    throw IOError("Failed to create directory " + featureDir);
  auto featureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(featurePath.c_str(), "wb"), &std::fclose);
  if(nullptr == featureFile)
    throw IOError("Failed to create feature file " + featurePath);
  size_t writecount = std::fwrite(&Dataset::FEATURE_HEADER, 4, 1, featureFile.get());
  if(1 != writecount)
    throw IOError("Failed to write to feature file " + featurePath);
  writecount = std::fwrite(features.data(), sizeof(MoveFeatures), features.size(), featureFile.get());
  if(features.size() != writecount)
    throw IOError("Failed to write to feature file " + featurePath);
  if(0 != std::fclose(featureFile.release()))
    throw IOError("Failed to write to feature file " + featurePath);
}
