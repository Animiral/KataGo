#include "program/strengthmodel.h"
#include "tests/tests.h"

using namespace std;

namespace {
bool fitsOneSample(vector<MoveFeatures> features, float target, int epochs, float weightPenalty, float learnrate, float& estimate);
}

namespace StrengthNetImpl {
void matmul(Tensor& y, const Tensor& W, const Tensor& x);
}

namespace Tests {
void runStrengthModelTests() {
  // string listFile = "games_labels_20.csv"; // shorter
  string listFile = "games_labels.csv";
  string featureDir = "featurecache";
  Dataset dataset;
  FeaturesAndTargets featuresTargets;

  try {
    dataset.load(listFile);
    featuresTargets = StrengthModel::getFeaturesAndTargetsCached(dataset, featureDir);
  }
  catch(const StringError& e) {
    cout << "skip strength model tests (" << e.what() << ")\n";
    return;
  }

  if(0) // disabled for calculation time
  {
    size_t sample = 47;
    cout << "- fits sample " << sample << " from list file " << listFile << ": ";

    float estimate;
    bool pass;
    float weightPenalty = 0;
    float learnrate = 0.01f;

    auto& fat = featuresTargets[sample];
    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(fat.first);
    net.setBatchSize(1);
    cout << fat.first.size() << " input features\n";
    // net.printWeights(cout, "initial values");
    // std::ofstream hfile("h_values.csv"); // debug csv

    int i;
    for(i = 0; i < 100; i++) {
      net.forward();
      // net.h.print(hfile, "h", false);
      net.backward(fat.second); //, 0);
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
    pass = fabs(net.getOutput() - fat.second) <= 0.1f;

    cout << "Estimate: " << estimate << ", target: " << fat.second << "\n";
    cout << (pass ? "pass" : "fail") << "\n";
  }

  if(0) // disabled for calculation time (1000 epochs per data sample)
  {
    cout << "- fits all samples from list file " << listFile << ": ";

    float estimate;
    bool pass;
    size_t upTo = min(featuresTargets.size(), 100ul); // speedup: cap samples to test

    pass = true;
    for(int i = 0; i < upTo; i++) {
      auto& fat = featuresTargets[i];
      if(!fitsOneSample(fat.first, fat.second, 1000, 0, 0.01f, estimate)) {
        pass = false;
        cerr << "Failed to fit sample " << i << " (" << fat.first.size() << " moves, target=" << fat.second << ", estimate=" << estimate << ")\n";
      }
    }

    cout << (pass ? "pass" : "fail") << "\n";
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

    for(int i = 0; i < epochs; i++) {
      net.forward();
      net.backward(target); //, 0);
      net.update(weightPenalty, learnrate);
    }

    net.forward();
    estimate = net.getOutput();
    return fabs(net.getOutput() - target) <= 0.1f;
}
}
