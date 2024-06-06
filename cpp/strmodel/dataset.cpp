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

void SelectedMoves::Moveset::insert(int index, Selection selection) {
  for(auto& m : moves)
    if(m.index == index)
      return;
  Move m;
  m.index = index;
  m.selection = selection;
  m.pla = C_EMPTY;
  m.pos = -1;
  moves.push_back(m);
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
      if(m.selection.trunk) before->selection.trunk = true;
      if(m.selection.pick) before->selection.pick = true;
      if(m.selection.head) before->selection.head = true;
      if(m.trunk) before->trunk = m.trunk;
      if(m.pick) before->pick = m.pick;
      if(m.head) before->head = m.head;
      if(m.pos >= 0) before->pos = m.pos;
    }
  }
}

bool SelectedMoves::Moveset::hasAllResults() const {
  auto needsResults = [](const Move& m) -> bool {
    return (m.selection.trunk && nullptr == m.trunk)
        || (m.selection.pick && nullptr == m.pick)
        || (m.selection.head && nullptr == m.head);
  };
  return moves.end() == std::find_if(moves.begin(), moves.end(), needsResults);
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
  return { { blackMoves, P_BLACK }, { whiteMoves, P_WHITE } };
}

namespace {

vector<int> getBufferOfMoveMember(
  const vector<SelectedMoves::Move>& moves,
  int SelectedMoves::Move::* member,
  size_t
) {
  vector<int> buffer(moves.size());
  for(size_t i = 0; i < moves.size(); i++) {
    buffer[i] = moves[i].*member;
  }
  return buffer;
}

pair<vector<float>, bool> getBufferOfMoveMember(
  const vector<SelectedMoves::Move>& moves,
  std::shared_ptr<vector<float>> SelectedMoves::Move::* member,
  size_t elsPerMove
) {
  vector<float> buffer(moves.size() * elsPerMove);
  bool haveData = false;
  for(size_t i = 0; i < moves.size(); i++) {
    const SelectedMoves::Move& move = moves[i];
    if(move.*member) {
      assert((move.*member)->size() == elsPerMove);
      std::copy((move.*member)->begin(), (move.*member)->end(), &buffer[i*elsPerMove]);
      haveData = true;
    }
  }
  return make_pair(move(buffer), haveData);
}


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
  if(!containingDir.empty() && !FileUtils::create_directories(containingDir))
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
  vector<int> indexBuffer = getBufferOfMoveMember(moves, &Move::index, 1);
  vector<int> posBuffer = getBufferOfMoveMember(moves, &Move::pos, 1);
  vector<float> trunkBuffer, pickBuffer, headBuffer;
  bool haveTrunks, havePicks, haveHeads;
  tie(trunkBuffer, haveTrunks) = getBufferOfMoveMember(moves, &Move::trunk, trunkSize);
  tie(pickBuffer, havePicks) = getBufferOfMoveMember(moves, &Move::pick, numTrunkFeatures);
  tie(headBuffer, haveHeads) = getBufferOfMoveMember(moves, &Move::head, numHeadFeatures);

  addFileToZip(*archive, indexBuffer, "index.bin");
  if(haveTrunks) 
    addFileToZip(*archive, trunkBuffer, "trunk.bin");
  if(havePicks) 
    addFileToZip(*archive, pickBuffer, "pick.bin");
  if(haveHeads) 
    addFileToZip(*archive, headBuffer, "head.bin");
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
    string trunkStr = m.trunk ? strprintf("trunk %f", vecChecksum(*m.trunk)) : "no trunk"s;
    string pickStr = m.pick ? strprintf("pick %f", vecChecksum(*m.pick)) : "no pick"s;
    string headStr = m.head ? strprintf("head(wr %f, pt %f, p %f, maxp %f, wr- %f, pt- %f)",
      m.head->at(0), m.head->at(1), m.head->at(2), m.head->at(3), m.head->at(4), m.head->at(5)) : "no head"s;
    stream << strprintf("%d: pos %d, %s, %s, %s\n", m.index, m.pos, trunkStr.c_str(), pickStr.c_str(), headStr.c_str());
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

  // find out how many positions/trunks are present in the archive
  zip_stat_t stat;
  if(0 != zip_stat_index(archive.get(), 0, 0, &stat))
    throw StringError("Error getting stat of first file in archive: "s + zip_strerror(archive.get()));
  uint64_t expectedCount = stat.size / sizeof(int);
  Moveset moveset{vector<Move>(expectedCount), pla};

  readFromZipPart(*archive, "index.bin", moveset, &Move::index, 1);
  if(zip_name_locate(archive.get(), "trunk.bin", ZIP_FL_ENC_RAW) >= 0)
    readFromZipPart(*archive, "trunk.bin", moveset, &Move::trunk, trunkSize);
  if(zip_name_locate(archive.get(), "pick.bin", ZIP_FL_ENC_RAW) >= 0)
    readFromZipPart(*archive, "pick.bin", moveset, &Move::pick, numTrunkFeatures);
  if(zip_name_locate(archive.get(), "head.bin", ZIP_FL_ENC_RAW) >= 0)
    readFromZipPart(*archive, "head.bin", moveset, &Move::head, numHeadFeatures);
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

pair<SelectedMoves::Moveset, size_t> findMovesOfColor(const string& sgfPath, Player pla, size_t capacity, Selection selection) {
  SelectedMoves::Moveset movesOfColor;
  movesOfColor.pla = pla;
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);
  size_t found = std::count_if(moves.begin(), moves.end(), [pla](Move m) { return pla == m.pla ; });
  size_t excess = found > capacity ? found - capacity : 0;
  found -= excess;

  for(int i = 0; i < moves.size(); i++) {
    if(pla == moves[i].pla) {
      if(excess > 0) {
        excess--;
        continue;
      }
      if(movesOfColor.moves.empty() || movesOfColor.moves.back().index != i)
        movesOfColor.insert(i, selection);
      if(selection.head) // for head features, we also need the board after the move to calculate winrate loss etc
        movesOfColor.insert(i+1, {false, false, selection.head});
    }
  }

  return std::make_pair(std::move(movesOfColor), found);

  // // size_t start = capacity > foundMoves.size() ? 0 : foundMoves.size() - capacity;

  // for(size_t i = start; i < foundMoves.size(); i++) {
  //   gameMoves.insert(foundMoves[i].index, foundMoves[i].selection);
  // }

  // return foundMoves.size() - start;
}

}

// size_t Dataset::getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize) const {
//   assert(player < players.size());
//   assert(game <= games.size());

//   // start from the game preceding the specified index
//   int gameIndex;
//   if(games.size() == game) {
//       gameIndex = players[player].lastOccurrence;
//   }
//   else {
//     const Game* gm = &games[game];
//     if(player == gm->black.player)
//       gameIndex = gm->black.prevGame;
//     else if(player == gm->white.player)
//       gameIndex = gm->white.prevGame;
//     else
//       gameIndex = static_cast<int>(game) - 1;
//   }

//   // go backwards in player's history and fill the buffer in backwards order
//   MoveFeatures* outptr = buffer + bufsize;
//   while(gameIndex >= 0 && outptr > buffer) {
//     while(gameIndex >= 0 && player != games[gameIndex].black.player && player != games[gameIndex].white.player)
//       gameIndex--; // this is just defense to ensure that we find a game which the player occurs in
//     if(gameIndex < 0)
//       break;
//     const Game* gm = &games[gameIndex];
//     bool isBlack = player == gm->black.player;
//     const auto& features = isBlack ? gm->black.features : gm->white.features;
//     for(int i = features.size(); i > 0 && outptr > buffer;)
//       *--outptr = features[--i];
//     gameIndex = isBlack ? gm->black.prevGame : gm->white.prevGame;
//   }

//   // if there are not enough features in history to fill the buffer, adjust
//   size_t count = bufsize - (outptr - buffer);
//   if(outptr > buffer)
//     std::memmove(buffer, outptr, count * sizeof(MoveFeatures));
//   return count;
// }

SelectedMoves Dataset::getRecentMoves(::Player player, size_t game, size_t capacity, Selection selection) const {
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
    pair<SelectedMoves::Moveset, size_t> moves_count = findMovesOfColor(historicGame.sgfPath, pla, capacity, selection);
    selectedMoves.bygame[historicGame.sgfPath] = std::move(moves_count.first);
    capacity -= moves_count.second;
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

