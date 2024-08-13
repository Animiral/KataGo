#include "strmodel/strengthmodel.h"
#include "strmodel/dataset.h"
#include "tests/tests.h"
#include "core/fileutils.h"
#include "core/global.h"
#include <iomanip>
#include "strmodel/using.h"

using namespace StrModel;

namespace {

using std::cout;

void createFeatureCache(const string& listFile, const string& featureDir);

}

namespace Tests {
void runStrengthModelTests(const string& listFile, const string& featureDir) {
  try {
    createFeatureCache(listFile, featureDir);
  }
  catch(const StringError& e) {
    cout << "skip strength model tests (" << e.what() << ")\n";
    return;
  }

  DatasetFiles files(featureDir);
  Dataset dataset(listFile, files);

  {
    cout << "- UT dataset contains " << dataset.games.size() << " games and " << dataset.players.size() << " players:\n";
    for(const Dataset::Game& game : dataset.games)
      cout << "\tGame " << game.sgfPath << " (" << "-TVBE"[game.set] << ") - " << game.black.features.size() << " black features, " << game.white.features.size() << " white features\n";
    for(const Dataset::Player& player : dataset.players)
      cout << "\tPlayer " << player.name << ", last occurred in game [" << player.lastOccurrence << "]\n";
  }

  {
    cout << "- StochasticPredictor.predict(): ";

    MoveFeatures blackFeatures[] = {{0, 0, 0, 0, 0, 2.f}, {0, 0, 0, 0, 0, 2.2f}, {0, 0, 0, 0, 0, 2.4f}};
    MoveFeatures whiteFeatures[] = {{0, 0, 0, 0, 0, 1.9f}, {0, 0, 0, 0, 0, 2.3f}, {0, 0, 0, 0, 0, 2.3f}};
    StochasticPredictor predictor;
    Dataset::Prediction prediction = predictor.predict(blackFeatures, 3, whiteFeatures, 3);
    // black mean = 2.2; white mean = 2.16666
    // black var = 0.04; white var = 0.053333
    // with 100 moves both, total black ploss mean = 3.3333; total black ploss stdev = sqrt(9.3333)
    float expectedScore = 0.137617f;
    bool pass = std::abs(prediction.score - expectedScore) < 0.0001;

    if(!pass)
      cout << "expected p=" << expectedScore << ", got " << prediction.score << "; ";

    cout << (pass ? "pass" : "fail") << "\n";
  }

  {
    cout << "- Dataset::randomSplit():\n";

    Rand rand(123ull); // reproducible seed
    Dataset dataset2 = dataset; // 3 games
    dataset2.games.insert(dataset2.games.end(), dataset.games.begin(), dataset.games.end()); // 3 more games
    dataset2.games.insert(dataset2.games.end(), dataset.games.begin(), dataset.games.end()); // 3 more games
    char setmarker[9];
    auto pickmarker = [](auto& game) { return "-TVBE"[game.set]; };

    cout << "    " << dataset2.games.size() << " games split 6/2/1: ";
    dataset2.randomSplit(rand, 0.66f, 0.22f);
    std::transform(dataset2.games.begin(), dataset2.games.end(), setmarker, pickmarker);
    for(size_t i = 0; i < 9; i++)
      cout << (i > 0 ? ", " : "") << setmarker[i];
    cout << "\n";

    cout << "    random 3-batch: ";
    dataset2.randomBatch(rand, 3);
    std::transform(dataset2.games.begin(), dataset2.games.end(), setmarker, pickmarker);
    for(size_t i = 0; i < 9; i++)
      cout << (i > 0 ? ", " : "") << setmarker[i];
    cout << "\n";

    cout << "    random 2-batch: ";
    dataset2.randomBatch(rand, 2);
    std::transform(dataset2.games.begin(), dataset2.games.end(), setmarker, pickmarker);
    for(size_t i = 0; i < 9; i++)
      cout << (i > 0 ? ", " : "") << setmarker[i];
    cout << "\n";

    cout << "    random 10-batch: ";
    dataset2.randomBatch(rand, 10);
    std::transform(dataset2.games.begin(), dataset2.games.end(), setmarker, pickmarker);
    for(size_t i = 0; i < 9; i++)
      cout << (i > 0 ? ", " : "") << setmarker[i];
    cout << "\n";
  }

}
}

namespace {

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
  const uint32_t FEATURE_HEADER_POC = 0xfea70235; // same as in strengthmodel.h
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
    if(                   1 != std::fwrite(&FEATURE_HEADER_POC, 4, 1, blackFeatureFile.get()) ||
       blackFeatures.size() != std::fwrite(blackFeatures.data(), sizeof(MoveFeatures), blackFeatures.size(), blackFeatureFile.get()) ||
                          1 != std::fwrite(&FEATURE_HEADER_POC, 4, 1, whiteFeatureFile.get()) ||
       whiteFeatures.size() != std::fwrite(whiteFeatures.data(), sizeof(MoveFeatures), whiteFeatures.size(), whiteFeatureFile.get()) ||
                          0 != std::fclose(blackFeatureFile.release()) ||
                          0 != std::fclose(whiteFeatureFile.release())) {
      throw IOError("Failed to create feature cache for test SGFs.");
    }
    std::getline(istrm, sgfPath); // throw away remaining line
  }
}

} // end namespace
