#include "program/strengthmodel.h"
#include "tests/tests.h"
#include "core/fileutils.h"
#include <iomanip>
#include "core/using.h"

using std::min;

namespace {

void printFeatures(const MoveFeatures* features, size_t count); // to cout
void createFeatureCache(const string& listFile, const string& featureDir);
bool fitsOneSample(vector<MoveFeatures> features, float target, int epochs, float weightPenalty, float learnrate, float& estimate);
bool featuresEqual(const MoveFeatures* actual, const MoveFeatures* expected, size_t count);

}

namespace StrengthNetImpl {
void matmul(Tensor& y, const Tensor& W, const Tensor& x);
}

namespace Tests {
void runStrengthModelTests(const string& modelFile, const string& listFile, const string& featureDir) {
  Dataset dataset;
  FeaturesAndTargets featuresTargets;

  StrengthModel StrengthModel(modelFile, nullptr, featureDir);

  try {
    createFeatureCache(listFile, featureDir);
    dataset.load(listFile, featureDir);
  }
  catch(const StringError& e) {
    cout << "skip strength model tests (" << e.what() << ")\n";
    return;
  }

  {
    cout << "- UT dataset contains " << dataset.games.size() << " games and " << dataset.players.size() << " players:\n";
    for(const Dataset::Game& game : dataset.games)
      cout << "\tGame " << game.sgfPath << " - " << game.blackFeatures.size() << " black features, " << game.whiteFeatures.size() << " white features\n";
    for(const Dataset::Player& player : dataset.players)
      cout << "\tPlayer " << player.name << ", last occurred in game [" << player.lastOccurrence << "]\n";
  }

  {
    cout << "- get recent moves: ";
    assert(dataset.players.size() >= 3 && dataset.games.size() >= 3);
    bool pass = true;
    vector<MoveFeatures> buffer(10);
    vector<MoveFeatures> expected(10);

    // if there are no previous features, we expect the buffer to be unchanged.
    buffer[0].winProb = 2;
    size_t count = dataset.getRecentMoves(2, 1, buffer.data(), 10);
    if(0 != count || 2 != buffer[0].winProb) {
      cout << "expected 0 features, got " << count << ": ";
      printFeatures(buffer.data(), count);
      pass = false;
    }

    // expect features in a sequence ordered from old to new
    count = dataset.getRecentMoves(0, 2, buffer.data(), 10);
    expected[0] = dataset.games[0].whiteFeatures[0]; // this makes assumptions about the UT dataset
    expected[1] = dataset.games[1].whiteFeatures[0];
    expected[2] = dataset.games[1].whiteFeatures[1];
    if(3 != count || !featuresEqual(buffer.data(), expected.data(), count)) {
      cout << "expected 3 features: ";
      printFeatures(expected.data(), 3);
      cout << ", got " << count << ": ";
      printFeatures(buffer.data(), count);
      pass = false;
    }

    // expect features cut off at the specified count, expect game index working 1 past end
    count = dataset.getRecentMoves(2, 3, buffer.data(), 2);
    expected[0] = dataset.games[2].blackFeatures[1];
    expected[1] = dataset.games[2].blackFeatures[2];
    if(2 != count || !featuresEqual(buffer.data(), expected.data(), count)) {
      cout << "expected 2 features: ";
      printFeatures(expected.data(), 2);
      cout << ", got " << count << ": ";
      printFeatures(buffer.data(), count);
      pass = false;
    }

    cout << (pass ? "pass" : "fail") << "\n";
  }

  if(0) // disabled for calculation time
  {
    size_t sample = 0;
    cout << "- fits sample " << sample << " from list file " << listFile << ": ";

    float estimate;
    bool pass;
    float weightPenalty = 0;
    float learnrate = 0.01f;

    auto& game = dataset.games[sample];
    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(game.blackFeatures);
    net.setBatchSize(1);
    cout << game.blackFeatures.size() << " input features\n";
    // net.printWeights(cout, "initial values");
    // std::ofstream hfile("h_values.csv"); // debug csv

    int i;
    for(i = 0; i < 100; i++) {
      net.forward();
      // net.h.print(hfile, "h", false);
      net.backward(game.blackRating); //, 0);
      net.update(weightPenalty, learnrate);
      cout << "epoch " << i << ": thetavar=" << net.thetaVar() << "\n";
    }

    // hfile.close();

    // // reconstruct the matrix multiplication
    // net.forward();
    // net.backward(fat.second); //, 0);
    // net.h_grad.print(cout, "h_grad (left)");
    // net.x.print(cout, "x (right)");
    // net.x.transpose();
    // StrengthNetImpl::matmul(net.W_grad, net.h_grad, net.x);
    // net.x.transpose();
    // net.W_grad.print(cout, "W_grad (result)");

    // GET THE FULL PICTURE
    // net.printWeights(cout, "after " + Global::intToString(i) + " epochs ");
    // net.forward();
    // net.backward(fat.second); //, 0);
    // net.update(weightPenalty, learnrate);
    // i++;
    // net.printState(cout, "after " + Global::intToString(i) + " epochs ");
    // net.printGrads(cout, "after " + Global::intToString(i) + " epochs ");
    // net.printWeights(cout, "after " + Global::intToString(i) + " epochs ");

    net.forward();
    estimate = net.getOutput();
    pass = fabs(net.getOutput() - game.blackRating) <= 0.1f;

    cout << "Estimate: " << estimate << ", target: " << game.blackRating << "\n";
    cout << (pass ? "pass" : "fail") << "\n";
  }

  if(0) // disabled for calculation time (1000 epochs per data sample)
  {
    cout << "- fits all samples from list file " << listFile << ": ";

    float estimate;
    bool pass;
    size_t upTo = min(dataset.games.size(), 100ul); // speedup: cap samples to test

    pass = true;
    for(int i = 0; i < upTo; i++) {
      auto& game = dataset.games[i];
      if(!fitsOneSample(game.blackFeatures, game.blackRating, 1000, 0, 0.01f, estimate)) {
        pass = false;
        cerr << "Failed to fit sample " << i << " (" << game.blackFeatures.size() << " moves, target=" << game.blackRating << ", estimate=" << estimate << ")\n";
      }
    }

    cout << (pass ? "pass" : "fail") << "\n";
  }
}
}

namespace {

void printFeatures(const MoveFeatures* features, size_t count) {
  cout << "[" << std::setprecision(2);
  for(auto* ft = features; ft < features + count; ft++) {
    if(features != ft)
      cout << ", ";
    cout << "(" << ft->winProb << "," << ft->lead << "," << ft->movePolicy << "," << ft->maxPolicy << "," << ft->winrateLoss << "," << ft->pointsLoss << ")";
  }
  cout << "]";
}

void mockGameFeatures(const string& sgfPath, vector<MoveFeatures>& blackFeaturesOut, vector<MoveFeatures>& whiteFeaturesOut) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  for(int turnIdx = 0; turnIdx < sgf->moves.size(); turnIdx++) {
    Move move = sgf->moves[turnIdx];
    // just make up some half plausible float values from move.loc
    MoveFeatures mf;
    mf.winProb = move.loc / 400.f;
    mf.lead = (move.loc - 200) / 10.f;
    mf.movePolicy = (500 - move.loc) / 500.f;
    mf.maxPolicy = (600 - move.loc) / 600.f;
    mf.winrateLoss = .1f;
    mf.pointsLoss = 2.f;
    (P_WHITE == move.pla ? whiteFeaturesOut : blackFeaturesOut).push_back(mf);
  }
}

void createFeatureCache(const string& listFile, const string& featureDir) {
  const uint32_t FEATURE_HEADER = 0xfea70235; // same as in strengthmodel.h
  std::ifstream istrm(listFile);
  string sgfPath; // we just assume that sgfPath is the first field in the CSV
  std::getline(istrm, sgfPath); // throw away header
  while (std::getline(istrm, sgfPath, ',')) {
    vector<MoveFeatures> blackFeatures, whiteFeatures;
    mockGameFeatures(sgfPath, blackFeatures, whiteFeatures);
    string blackFeaturePath = Global::strprintf("%s/%s_%sFeatures.bin", featureDir.c_str(), Global::chopSuffix(sgfPath, ".sgf").c_str(), PlayerIO::playerToString(P_BLACK).c_str());
    string whiteFeaturePath = Global::strprintf("%s/%s_%sFeatures.bin", featureDir.c_str(), Global::chopSuffix(sgfPath, ".sgf").c_str(), PlayerIO::playerToString(P_WHITE).c_str());
    FileUtils::create_directories(FileUtils::dirname(blackFeaturePath)); // ensure dir structure
    auto blackFeatureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(blackFeaturePath.c_str(), "wb"), &std::fclose);
    auto whiteFeatureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(whiteFeaturePath.c_str(), "wb"), &std::fclose);
    if(                   1 != std::fwrite(&FEATURE_HEADER, 4, 1, blackFeatureFile.get()) ||
       blackFeatures.size() != std::fwrite(blackFeatures.data(), sizeof(MoveFeatures), blackFeatures.size(), blackFeatureFile.get()) ||
                          1 != std::fwrite(&FEATURE_HEADER, 4, 1, whiteFeatureFile.get()) ||
       whiteFeatures.size() != std::fwrite(whiteFeatures.data(), sizeof(MoveFeatures), whiteFeatures.size(), whiteFeatureFile.get()) ||
                          0 != std::fclose(blackFeatureFile.release()) ||
                          0 != std::fclose(whiteFeatureFile.release())) {
      throw IOError("Failed to create feature cache for test SGFs.");
    }
    std::getline(istrm, sgfPath); // throw away remaining line
  }
}

bool fitsOneSample(vector<MoveFeatures> features, float target, int epochs, float weightPenalty, float learnrate, float& estimate) {
    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(features);
    net.setBatchSize(1);

    for(int i = 0; i < epochs; i++) {
      net.forward();
      net.backward(target); //, 0);
      net.update(weightPenalty, learnrate);
    }

    net.forward();
    estimate = net.getOutput();
    return fabs(net.getOutput() - target) <= 0.1f;
}

bool featuresEqual(const MoveFeatures* actual, const MoveFeatures* expected, size_t count) {
  for(size_t i = 0; i < count; i++) {
    if(actual[i].winProb != expected[i].winProb || 
       actual[i].lead != expected[i].lead ||
       actual[i].movePolicy != expected[i].movePolicy ||
       actual[i].maxPolicy != expected[i].maxPolicy ||
       actual[i].winrateLoss != expected[i].winrateLoss ||
       actual[i].pointsLoss != expected[i].pointsLoss) {
      return false;
    }
  }
  return true;
}

} // end namespace
