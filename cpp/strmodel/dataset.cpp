#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include <zip.h>
#include "strmodel/dataset.h"
#include "game/board.h"
#include "dataio/sgf.h"
#include "core/global.h"
#include "core/fileutils.h"
#include "core/using.h"

using std::unique_ptr;
using std::map;
using std::move;
using namespace std::literals;
using Global::strprintf;

void SelectedMoves::Moveset::insert(int index, Player pla) {
  for(auto& m : moves)
    if(m.index == index)
      return;
  moves.push_back({index, pla, nullptr, nullptr, -1});
}

void SelectedMoves::Moveset::merge(const SelectedMoves::Moveset& rhs) {
  auto before = moves.begin();
  for(const Move& m : rhs.moves) {
    while(before < moves.end() && before->index < m.index)
      before++; // find next place to insert move in ascending order

    if(moves.end() == before || before->index != m.index) { // no duplicate inserts
      before = ++moves.insert(before, m);
    }
    if(moves.end() != before && before->index == m.index) { // in doubt, pick rhs data
      *before = m;
    }
  }
}

bool SelectedMoves::Moveset::hasAllPicks() const {
  return moves.end() == std::find_if(moves.begin(), moves.end(), [](const Move& m) { return nullptr == m.pick; });
}

void SelectedMoves::Moveset::releaseStorage() {
  for(Move& move : moves){
    move.trunk.reset();
    move.pick.reset();
  }
}

pair<SelectedMoves::Moveset, SelectedMoves::Moveset> SelectedMoves::Moveset::splitBlackWhite() const {
  vector<Move> blackMoves, whiteMoves;
  std::copy_if(moves.begin(), moves.end(), std::back_inserter(blackMoves), [](const Move& m){ return P_BLACK == m.pla; });
  std::copy_if(moves.begin(), moves.end(), std::back_inserter(whiteMoves), [](const Move& m){ return P_WHITE == m.pla; });
  return { { blackMoves }, { whiteMoves } };
}

namespace {

template<typename T>
void addFileToZip(zip_t& archive, const vector<T>& buffer, const char* name) {
  unique_ptr<zip_source_t, decltype(&zip_source_free)> source{
    zip_source_buffer(&archive, buffer.data(), buffer.size()*sizeof(T), 0),
    &zip_source_free
  };
  if(!source)
    throw StringError("Error creating zip source: "s + zip_strerror(&archive));

  if(zip_add(&archive, name, source.get()) < 0)
    throw StringError(strprintf("Error adding %s to zip archive: %s", name, zip_strerror(&archive)));
  source.release(); // after zip_add, source is managed by libzip
}

}

void SelectedMoves::Moveset::writeToZip(const string& filePath) const {
  string containingDir = FileUtils::dirname(filePath);
  if(!FileUtils::create_directories(containingDir))
    throw IOError("Failed to create directory " + containingDir);

  int err;
  unique_ptr<zip_t, decltype(&zip_discard)> archive{
    zip_open(filePath.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err),
    &zip_discard
  };
  if(!archive) {
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    string errstr = zip_error_strerror(&error);
    zip_error_fini(&error);
    throw StringError("Error opening zip archive: "s + errstr);
  }

  // merge individual moves data into contiguous buffers
  vector<int> indexBuffer(moves.size());
  vector<float> trunkBuffer(moves.size() * trunkSize);
  vector<float> pickBuffer(moves.size() * numTrunkFeatures);
  vector<int> posBuffer(moves.size());
  bool haveTrunks = false;
  bool havePicks = false;
  for(size_t i = 0; i < moves.size(); i++) {
    const Move& move = moves[i];
    indexBuffer[i] = move.index;
    if(move.trunk) {
      assert(move.trunk->size() == trunkSize);
      std::copy(move.trunk->begin(), move.trunk->end(), &trunkBuffer[i*trunkSize]);
      haveTrunks = true;
    }
    if(move.pick) {
      assert(move.pick->size() == numTrunkFeatures);
      std::copy(move.pick->begin(), move.pick->end(), &pickBuffer[i*numTrunkFeatures]);
      havePicks = true;
    }
    posBuffer[i] = move.pos;
  }

  addFileToZip(*archive, indexBuffer, "index.bin");
  if(haveTrunks) 
    addFileToZip(*archive, trunkBuffer, "trunk.bin");
  if(havePicks) 
    addFileToZip(*archive, pickBuffer, "pick.bin");
  addFileToZip(*archive, posBuffer, "movepos.bin");

  zip_t* archivep = archive.release();
  if(zip_close(archivep) != 0) {
    StringError error("Error writing zip archive: "s + zip_strerror(archivep));
    zip_discard(archivep);
    throw error;
  }
}

namespace {

// condense vec to a single printable number, likely different from vecs (trunks) of other positions,
// but also close in value to very similar vecs (tolerant of float inaccuracies)
float vecChecksum(const vector<float>& vec) {
  float sum = 0.0f;
  float sos = 0.0f;
  float weight = 1.0f;
  float decay = 0.9999667797285222f; // = pow(0.01, (1/(vec.size()-1))) -> smallest weight is 0.01

  for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i] * weight;
      sos += vec[i] * vec[i];
      weight *= decay;
  }

  return sum + std::sqrt(sos);
}

}

void SelectedMoves::Moveset::printSummary(std::ostream& stream) const {
  for(const Move& m : moves) {
    float trunkSum = m.trunk ? vecChecksum(*m.trunk) : 0;
    float pickSum = m.pick ? vecChecksum(*m.pick) : 0;
    stream << strprintf("%d: pos %d, trunk %f, pick %f\n", m.index, m.pos, trunkSum, pickSum);
  }
}

namespace {

void readMoveMemberFromZip(
  zip_file_t& file,
  SelectedMoves::Move& move,
  int SelectedMoves::Move::* member,
  size_t
) {
  int64_t read = zip_fread(&file, &(move.*member), sizeof(int));
  if(sizeof(int) != read)
    throw StringError(strprintf("Error reading zipped file data: %s", zip_file_strerror(&file)));
}

void readMoveMemberFromZip(
  zip_file_t& file,
  SelectedMoves::Move& move,
  std::shared_ptr<vector<float>> SelectedMoves::Move::* member,
  size_t elsPerMove
) {
  auto& storage = move.*member;
  storage.reset(new vector<float>(elsPerMove));
  int64_t read = zip_fread(&file, storage->data(), elsPerMove*sizeof(float));
  if(elsPerMove*sizeof(float) != read)
    throw StringError(strprintf("Error reading zipped file data: %s", zip_file_strerror(&file)));
}

template<typename DataMemberType>
uint64_t expectedSize(size_t moves, size_t elsPerMove) = delete;
template<>
uint64_t expectedSize<int>(size_t moves, size_t elsPerMove) {
  assert(1 == elsPerMove); // we can only fit one int per move
  return moves * sizeof(int);
}
template<>
uint64_t expectedSize<std::shared_ptr<vector<float>>>(size_t moves, size_t elsPerMove) {
  return moves * elsPerMove * sizeof(float);
}

// extract one file in the zip, run checks, and extract each stored element to its appropriate move with readMoveMemberFromZip
template<typename T>
void readFromZipPart(
  zip_t& archive,
  const char* name,
  SelectedMoves::Moveset& moveset,
  T SelectedMoves::Move::* member,
  size_t elsPerMove
) {
  int64_t index = zip_name_locate(&archive, name, ZIP_FL_ENC_RAW);
  if(index < 0)
      throw StringError(strprintf("File %s not found in archive.", name));

  zip_stat_t stat;
  if(0 != zip_stat_index(&archive, index, 0, &stat))
    throw StringError(strprintf("Error getting %s file information: %s", name, zip_strerror(&archive)));
  uint64_t expected = expectedSize<T>(moveset.moves.size(), elsPerMove);
  if(stat.size != expected)
    throw StringError(strprintf("%s data has %d bytes, but expected %d bytes", name, stat.size, expected));

  unique_ptr<zip_file_t, decltype(&zip_fclose)> file{
    zip_fopen_index(&archive, index, ZIP_RDONLY),
    &zip_fclose
  };
  if(!file)
    throw StringError(strprintf("Error opening %s in zip archive: %s", name, zip_strerror(&archive)));

  for(SelectedMoves::Move& move : moveset.moves)
    readMoveMemberFromZip(*file, move, member, elsPerMove);
}

}

SelectedMoves::Moveset SelectedMoves::Moveset::readFromZip(const string& filePath, Player pla) {
  int err;
  unique_ptr<zip_t, decltype(&zip_close)> archive{
    zip_open(filePath.c_str(), ZIP_RDONLY, &err),
    &zip_close
  };
  if(!archive) {
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    string errstr = zip_error_strerror(&error);
    zip_error_fini(&error);
    throw StringError("Error opening zip archive: "s + errstr);
  }

  int countEntries = zip_get_num_entries(archive.get(), 0);
  if(3 != countEntries)
    throw StringError(strprintf("Expected exactly three files in the archive, got %d.", countEntries));

  // find out how many positions/trunks are present in the archive
  zip_stat_t stat;
  if(0 != zip_stat_index(archive.get(), 0, 0, &stat))
    throw StringError("Error getting stat of first file in archive: "s + zip_strerror(archive.get()));
  uint64_t expectedCount = stat.size / sizeof(int);
  Moveset moveset{vector<Move>(expectedCount)};

  readFromZipPart(*archive, "index.bin", moveset, &Move::index, 1);
  readFromZipPart(*archive, "trunk.bin", moveset, &Move::trunk, trunkSize);
  readFromZipPart(*archive, "pick.bin", moveset, &Move::pick, trunkSize);
  readFromZipPart(*archive, "movepos.bin", moveset, &Move::pos, 1);

  std::for_each(moveset.moves.begin(), moveset.moves.end(), [pla](Move& m) { m.pla = pla; });
  return moveset;
}

size_t SelectedMoves::size() const {
  auto addSize = [](size_t a, const std::pair<string, Moveset>& kv) { return a + kv.second.moves.size(); };
  return std::accumulate(bygame.begin(), bygame.end(), size_t(0), addSize);
}

void SelectedMoves::merge(const SelectedMoves& rhs) {
  for(auto kv : rhs.bygame) {
    Moveset& mset = bygame[kv.first];
    mset.merge(kv.second);
  }
}

void SelectedMoves::copyFeaturesFrom(const SelectedMoves& rhs) {
  for(auto& kv : bygame) {
    vector<Move>& mymoves = kv.second.moves;
    const vector<Move>& rmoves = rhs.bygame.at(kv.first).moves;
    // both mymoves and rmoves are ordered by move index
    size_t rindex = 0;
    for(size_t i = 0; i < mymoves.size(); i++) {
      while(rindex < rmoves.size() && rmoves[rindex].index != mymoves[i].index)
        rindex++;
      if(rindex >= rmoves.size())
        throw StringError(strprintf("Game %s move %d missing from precomputed data.", kv.first.c_str(), mymoves[i].index));

      mymoves[i].trunk = rmoves[rindex].trunk;
      mymoves[i].pick = rmoves[rindex].pick;
      mymoves[i].pos = rmoves[rindex].pos;
    }
  }
}

void Dataset::load(const string& path, const string& featureDir) {
  std::ifstream istrm(path);
  if (!istrm.is_open())
    throw IOError("Could not read dataset from " + path);

  std::string line;
  std::getline(istrm, line);
  if(!istrm)
    throw IOError("Could not read header line from " + path);
  line = Global::trim(line);

  // clean any previous data
  games.clear();
  players.clear();
  nameIndex.clear();

  // map known fieldnames to row indexes, wherever they may be
  enum class F { ignore, sgfPath, whiteName, blackName, whiteRating, blackRating, score, predictedScore, set };
  vector<F> fields;
  std::string field;
  std::istringstream iss(line);
  while(std::getline(iss, field, ',')) {
    if("File" == field) fields.push_back(F::sgfPath);
    else if("Player White" == field) fields.push_back(F::whiteName);
    else if("Player Black" == field) fields.push_back(F::blackName);
    else if("WhiteRating" == field) fields.push_back(F::whiteRating);
    else if("BlackRating" == field) fields.push_back(F::blackRating);
    else if("Winner" == field || "Judgement" == field || "Score" == field) fields.push_back(F::score);
    else if("PredictedScore" == field) fields.push_back(F::predictedScore);
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
      case F::whiteRating:
        game.white.rating = Global::stringToFloat(field);
        break;
      case F::blackRating:
        game.black.rating = Global::stringToFloat(field);
        break;
      case F::score:
        if('b' == field[0] || 'B' == field[0])
          game.score = 1;
        else if('w' == field[0] || 'W' == field[0])
          game.score = 0;
        else
          game.score = std::strtof(field.c_str(), nullptr);
        break;
      case F::predictedScore:
        game.prediction.score = std::strtof(field.c_str(), nullptr);
        break;
      case F::set:
        if("-" == field) game.set = Game::none;
        if("t" == field || "T" == field) game.set = Game::training;
        if("v" == field || "V" == field) game.set = Game::validation;
        if("b" == field || "B" == field) game.set = Game::batch;
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
    loadPocFeatures(featureDir);
}

namespace {
  const char* scoreToString(float score) {
    // only 3 values are really allowed, all perfectly representable in float
    if(0 == score)     return "0";
    if(1 == score)     return "1";
    if(0.5 == score)   return "0.5";
    else               return "(score error)";
  }
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
    int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%s,%.2f,%.2f,%.9f,%f,%f,%c\n",
      game.sgfPath.c_str(), whiteName.c_str(), blackName.c_str(),
      scoreToString(game.score), game.black.rating, game.white.rating,
      game.prediction.score, game.prediction.blackRating, game.prediction.whiteRating, "-TVBE"[game.set]);
    if(printed <= 0)
      throw IOError("Error during formatting.");
    ostrm << buffer.get();
  }

  ostrm.close();
}

namespace {

int countMovesOfColor(const string& sgfPath, Player pla) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  return std::count_if(moves.begin(), moves.end(), [pla](Move m) { return pla == m.pla; });
}

int findMovesOfColor(const string& sgfPath, Player pla, SelectedMoves& selectedMoves, size_t capacity) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);
  vector<int> foundMoves;

  for(int i = 0; i < moves.size(); i++) {
    if(pla == moves[i].pla) {
      foundMoves.push_back(i);
    }
  }

  // insert only up to capacity, with preference for later moves
  auto& gameMoves = selectedMoves.bygame[sgfPath];
  size_t start = capacity > foundMoves.size() ? 0 : foundMoves.size() - capacity;

  for(size_t i = start; i < foundMoves.size(); i++) {
    gameMoves.insert(foundMoves[i], pla);
  }

  return foundMoves.size() - start;
}

}

size_t Dataset::getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize) const {
  assert(player < players.size());
  assert(game <= games.size());

  // start from the game preceding the specified index
  int gameIndex;
  if(games.size() == game) {
      gameIndex = players[player].lastOccurrence;
  }
  else {
    const Game* gm = &games[game];
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
    const Game* gm = &games[gameIndex];
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

SelectedMoves Dataset::getRecentMoves(::Player player, size_t game, size_t capacity) const {
  SelectedMoves selectedMoves;
  const Game& gameData = games[game];
  auto& info = P_BLACK == player ? gameData.black : gameData.white;
  // Traverse game history
  size_t playerId = info.player;
  int historic = info.prevGame; // index of prev game

  while(0 < capacity && historic >= 0) {
    const Dataset::Game& historicGame = games[historic];

    ::Player pla;
    if(playerId == historicGame.black.player) {
      pla = P_BLACK;
      historic = historicGame.black.prevGame;
    } else if(playerId == historicGame.white.player) {
      pla = P_WHITE;
      historic = historicGame.white.prevGame;
    } else {
      throw StringError(strprintf("Game %s does not contain player %d (name=%s)",
        historicGame.sgfPath.c_str(), playerId, players[playerId].name.c_str()));
    }
    capacity -= findMovesOfColor(historicGame.sgfPath, pla, selectedMoves, capacity);
  }

  return selectedMoves;
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

void Dataset::markRecentGames(int windowSize, Logger* logger) {
	vector<int> recentGames; // indexes into this->games

  for(Game& game : games) {
    if(Game::none == game.set)
      continue;

    if(logger)
      logger->write("Mark recent moves of " + game.sgfPath + " ...");
    for(auto& info : {game.black, game.white}) {
      // Traverse game history
      int idx = 0;
      size_t playerId = info.player;
      int historic = info.prevGame; // index of prev game

      while(idx < windowSize && historic >= 0) {
        recentGames.push_back(historic);
        const Dataset::Game& historicGame = games[historic];

        ::Player pla;
        if(playerId == historicGame.black.player) {
          pla = P_BLACK;
          historic = historicGame.black.prevGame;
        } else if(playerId == historicGame.white.player) {
          pla = P_WHITE;
          historic = historicGame.white.prevGame;
        } else {
          throw StringError(strprintf("Game %s does not contain player %d (name=%s)",
            historicGame.sgfPath.c_str(), playerId, players[playerId].name.c_str()));
        }
        idx += countMovesOfColor(historicGame.sgfPath, pla);
      }
    }
  }

  for(int index : recentGames) {
    games[index].set = Game::batch;
  }
}

const uint32_t Dataset::FEATURE_HEADER = 0xfea70236;
const uint32_t Dataset::FEATURE_HEADER_POC = 0xfea70235;

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

void Dataset::loadPocFeatures(const std::string& featureDir) {
  for(Game& game : games) {
    string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
    string blackFeaturesPath = strprintf("%s/%s_BlackFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    string whiteFeaturesPath = strprintf("%s/%s_WhiteFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    game.black.features = readPocFeaturesFromFile(blackFeaturesPath);
    game.white.features = readPocFeaturesFromFile(whiteFeaturesPath);
  }
}

vector<MoveFeatures> Dataset::readPocFeaturesFromFile(const string& featurePath) {
  vector<MoveFeatures> features;
  auto featureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(featurePath.c_str(), "rb"), &std::fclose);
  if(nullptr == featureFile)
    throw IOError("Failed to read access feature file " + featurePath);
  uint32_t header; // must match
  size_t readcount = std::fread(&header, 4, 1, featureFile.get());
  if(1 != readcount || FEATURE_HEADER_POC != header)
    throw IOError("Failed to read from feature file " + featurePath);
  while(!std::feof(featureFile.get())) {
    MoveFeatures mf;
    readcount = std::fread(&mf, sizeof(MoveFeatures), 1, featureFile.get());
    if(1 == readcount)
      features.push_back(mf);
  }
  return features;
}

