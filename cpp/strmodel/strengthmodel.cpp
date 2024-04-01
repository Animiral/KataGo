#include "strengthmodel.h"
#include "core/global.h"
#include "core/fileutils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include "core/using.h"

using std::sqrt;

float Predictor::glickoScore(float blackRating, float whiteRating) {
  float GLICKO2_SCALE = 173.7178f;
  return 1 / (1 + exp((whiteRating - blackRating) / GLICKO2_SCALE));
  // Elo Score (for reference):
  // float Qblack = static_cast<float>(std::pow(10, blackRating / 400));
  // float Qwhite = static_cast<float>(std::pow(10, whiteRating / 400));
  // return Qblack / (Qblack + Qwhite);
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

// Because of float limitations, normcdf(x) maxes out for |x| > 5.347.
// Therefore its value is capped such that the result P as well as
// 1.f-P are in the closed interval (0, 1) under float arithmetic.
float normcdf(float x) noexcept {
  float P = .5f * (1.f + std::erf(x / sqrt(2.f)));
  if(P >= 1) return std::nextafter(1.f, 0.f);     // =0.99999994f, log(0.99999994f): -5.96e-08
  if(P <= 0) return 1 - std::nextafter(1.f, 0.f); // =0.00000006f, log(0.00000006f): -16.63
  else return P;
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
  float z = sqrt(gamelength) * (wplavg - bplavg) / sqrt(bplvar + wplvar + epsilon); // white pt advantage in standard normal distribution at move# [2*gamelength]
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
  prediction.score = glickoScore(prediction.blackRating, prediction.whiteRating);
  return prediction;
}


StrengthModel::StrengthModel(const string& strengthModelFile, Dataset* dataset_, Rand* rand) noexcept
  : net(), dataset(dataset_)
{
  bool loaded = false;
  if(!strengthModelFile.empty()) {
    try {
      net.loadModelFile(strengthModelFile);
      loaded = true;
    }
    catch(const IOError& error) {
      cerr << Global::strprintf("Could not load existing strength model from %s: %s Random-initializing new strength model.\n",
         strengthModelFile.c_str(), error.what());
    }
  }
  if(!loaded) {
    if(rand) {
      net.randomInit(*rand);
    }
    else {
      Rand localRand;
      net.randomInit(localRand);
    }
  }
}

void StrengthModel::extractFeatures(const std::string& featureDir, const Search& search, Logger* logger) {
  for(Dataset::Game& game : dataset->games) {
    string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
    string blackFeaturesPath = Global::strprintf("%s/%s_BlackFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());

    if(FileUtils::exists(blackFeaturesPath) && FileUtils::exists(blackFeaturesPath))
      continue; // skip this game as it has already been analyzed

    if(logger)
      logger->write("Extracting from " + game.sgfPath + "...");

    auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(game.sgfPath));
    extractGameFeatures(*sgf, search, game.black.features, game.white.features);
    if(!FileUtils::exists(blackFeaturesPath))
      writeFeaturesToFile(blackFeaturesPath, game.black.features);
    if(!FileUtils::exists(whiteFeaturesPath))
      writeFeaturesToFile(whiteFeaturesPath, game.white.features);
  }
}

void StrengthModel::train(int epochs, int steps, size_t batchSize, float weightPenalty, float learnrate, size_t windowSize, Rand& rand) {
  SmallPredictor predictor(net);

  {
    // evaluate starting net
    Evaluation trainingEval = evaluate(predictor, Dataset::Game::training, windowSize);
    Evaluation validationEval = evaluate(predictor, Dataset::Game::validation, windowSize);
    float theta_var = net.thetaVar();
    // cout << "Epoch " << e << ": mse=" << std::fixed << std::setprecision(3) << mse << "\n";
    cout << Global::strprintf("Before training: sqrt_mse_T=%.2f, alpha_T=%.3f, lbd_T=%.2f, sqrt_mse_V=%.2f, alpha_V=%.3f, lbd_V=%.2f, theta^2=%.4f\n",
      sqrt(trainingEval.mse), trainingEval.rate, trainingEval.logp,
      sqrt(validationEval.mse), validationEval.rate, validationEval.logp,
      theta_var);
  }

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
    cout << Global::strprintf("Epoch %d: sqrt_mse_T=%.2f, alpha_T=%.3f, lbd_T=%.2f, sqrt_mse_V=%.2f, alpha_V=%.3f, lbd_V=%.2f, theta^2=%.4f, grad^2=%.4f\n",
      e, sqrt(trainingEval.mse), trainingEval.rate, trainingEval.logp,
      sqrt(validationEval.mse), validationEval.rate, validationEval.logp,
      theta_var, grads_var);
  }
}

StrengthModel::Evaluation StrengthModel::evaluate(Predictor& predictor, int set, size_t windowSize) {
  vector<MoveFeatures> blackFeatures(windowSize);
  vector<MoveFeatures> whiteFeatures(windowSize);
  size_t successCount = 0;
  size_t count = 0;
  float mse = 0;
  float logp = 0;

  for(size_t i = 0; i < dataset->games.size(); i++) {
    Dataset::Game& gm = dataset->games[i];

    // always perform prediction, even for games not in the set
    size_t blackCount = dataset->getRecentMoves(gm.black.player, i, blackFeatures.data(), windowSize);
    size_t whiteCount = dataset->getRecentMoves(gm.white.player, i, whiteFeatures.data(), windowSize);
    gm.prediction = predictor.predict(blackFeatures.data(), blackCount, whiteFeatures.data(), whiteCount);

    // skip games not in set, also set < 0 means "include everything"
    if(set >= 0 && gm.set != set && !(Dataset::Game::training == set && Dataset::Game::batch == gm.set))
      continue;

    float diffBlack = gm.black.rating - gm.prediction.blackRating;
    float diffWhite = gm.white.rating - gm.prediction.whiteRating;
    mse += diffBlack * diffBlack + diffWhite * diffWhite;
    if((.5 >= gm.score && .5 >= gm.prediction.score)  // white win predicted (0.5 counts as prediction for white)
    || (.5 < gm.score && .5 < gm.prediction.score)) { // black win predicted
      successCount++;
    }
    logp += std::log(1 - std::abs(gm.score - gm.prediction.score));
    count++;
  }

  float rate = float(successCount) / count;
  mse /= count;
  logp /= count;
  return { count, mse, rate, logp };
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

  net.setInput({playerFeatures});
  net.forward();
  vector<float> rating = net.getOutput();

  Analysis analysis;
  for(const auto& mf : playerFeatures) {
    analysis.avgWRLoss += mf.winrateLoss;
    analysis.avgPLoss += mf.pointsLoss;
  }
  size_t N = playerFeatures.size();
  analysis.moveCount = N;
  analysis.avgWRLoss /= N;
  analysis.avgPLoss /= N;
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
  size_t writecount = std::fwrite(&Dataset::FEATURE_HEADER_POC, 4, 1, featureFile.get());
  if(1 != writecount)
    throw IOError("Failed to write to feature file " + featurePath);
  writecount = std::fwrite(features.data(), sizeof(MoveFeatures), features.size(), featureFile.get());
  if(features.size() != writecount)
    throw IOError("Failed to write to feature file " + featurePath);
  if(0 != std::fclose(featureFile.release()))
    throw IOError("Failed to write to feature file " + featurePath);
}
