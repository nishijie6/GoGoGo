"""Training-only weakness mining preserves the actual move history."""
import json
from pathlib import Path
import tempfile
import unittest

from weiqi.rl_targeted_positions import mine_game, mine_training_files
from weiqi.engine import GoGame


def annotated_game(suffix=''):
    moves = []
    for i in range(40):
        xy = chr(97 + i % 19) + chr(97 + i // 19)
        probability = .7 if i == 31 else .5
        moves.append(f';{"B" if i % 2 == 0 else "W"}[{xy}]C[{probability} {1-probability} 0.00 0 v=400 weight=1]')
    return ('(;FF[4]SZ[19]KM[6.5]RU[koPOSITIONALscoreAREAtaxNONEsui0]RE[B+1.5]'
            f'C[{suffix}]' + ''.join(moves) + ')')


class TargetedPositionTests(unittest.TestCase):
    def test_mining_keeps_history_instead_of_reconstructing_a_board(self):
        position = mine_game(annotated_game())
        self.assertEqual(position['turn'], 32)
        self.assertEqual(len(position['moveLocs']), 32)
        self.assertEqual(position['hintLoc'], 'null')
        self.assertEqual(position['moveLocs'][0], 'A19')
        self.assertEqual(position['moveLocs'][19], 'A18')
        state = GoGame(19, komi=6.5)
        for i in range(32):
            self.assertTrue(state.play(*divmod(i, 19)).legal)
        self.assertEqual(position['movePlas'][-1], 'W')

    def test_wrong_rules_or_initial_setup_are_not_mined(self):
        game = annotated_game()
        for changed in (game.replace('KM[6.5]', 'KM[7.5]'),
                        game.replace('koPOSITIONAL', 'koSIMPLE'),
                        game.replace('SZ[19]', 'SZ[19]AB[dd]')):
            self.assertIsNone(mine_game(changed))
        self.assertIsNone(mine_game(game, minimum_swing=.8))

    def test_duplicates_and_whole_game_partitions_are_separated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training.sgfs'
            games = [annotated_game(str(i)) for i in range(200)]
            path.write_text('\n'.join(games + games), encoding='utf8')
            train, val, rejected = mine_training_files([path], limit=80)
            train_ids = {p['game_sha256'] for p in train}
            val_ids = {p['game_sha256'] for p in val}
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(len(train_ids), len(train))
            self.assertGreater(len(val), 10)
            self.assertEqual(rejected, 0)
            json.dumps(train)


if __name__ == '__main__':
    unittest.main()
