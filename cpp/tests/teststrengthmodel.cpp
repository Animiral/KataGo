#include "program/strengthmodel.h"
#include "tests/tests.h"

using namespace std;

namespace {
bool fitsOneSample(vector<MoveFeatures> features, float target, int epochs, float weightPenalty, float learnrate, float& estimate);
}

namespace Tests {
void runStrengthModelTests() {

  if(0) // disabled for calculation time (1000 epochs per data sample)
  {
    string listFile = "games_labels.csv";
    string featureDir = "featurecache";
    cout << "- fits all samples from list file " << listFile << ": ";

    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    Dataset dataset;
    FeaturesAndTargets featuresTargets;
    float estimate;
    bool pass;

    try {
      dataset = StrengthModel::loadDataset(listFile);
      featuresTargets = StrengthModel::getFeaturesAndTargetsCached(dataset, featureDir);
    }
    catch(const StringError& e) {
      cout << "skip (" << e.what() << ")\n";
      goto skip_test_fitall;
    }

    pass = true;
    for(int i = 0; i < featuresTargets.size(); i++) {
      auto& fat = featuresTargets[i];
      if(!fitsOneSample(fat.first, fat.second, 1000, 0, 0.01f, estimate)) {
        pass = false;
        cerr << "Failed to fit sample " << i << " (" << fat.first.size() << " moves, target=" << fat.second << ", estimate=" << estimate << ")\n";
      }
    }

    cout << (pass ? "pass" : "fail") << "\n";
    skip_test_fitall:;
  }
}
}

namespace {
bool fitsOneSample(vector<MoveFeatures> features, float target, int epochs, float weightPenalty, float learnrate, float& estimate) {
    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(features);
    net.setBatchSize(1);

    for(int i = 0; i < epochs; i++) { // perfectly fit to threemoves input
      net.forward();
      net.backward(target); //, 0);
      net.update(weightPenalty, learnrate);
    }

    net.forward();
    estimate = net.getOutput();
    return fabs(net.getOutput() - target) <= 0.1f;
}
}
