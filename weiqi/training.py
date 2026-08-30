"""Curated joseki and life-and-death reasoning lessons.

The lesson data is deliberately independent from Tkinter so it can be tested,
reused, and extended without touching the user interface.  Coordinates are
zero based ``(row, column)`` just like :mod:`weiqi.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .engine import BLACK, EMPTY, WHITE, BoardHash, Point, opponent


JOSEKI_CATEGORY = "常见定式"
LIFE_AND_DEATH_CATEGORY = "死活题"
TRAINING_CATEGORIES = (JOSEKI_CATEGORY, LIFE_AND_DEATH_CATEGORY)

JOSEKI_NOTICE = (
    "定式是局部双方大体可接受的参考次序，不是全盘唯一答案。实际选择还要看邻角、"
    "外围子力、征子和行棋方向；训练重点是理解每手目的，而不是死背。"
)


@dataclass(frozen=True)
class TrainingMove:
    """One expected move and the explanation revealed after it is played."""

    color: int
    point: Point
    name: str
    prompt: str
    hint: str
    reason: str


@dataclass(frozen=True)
class TrainingLesson:
    """A guided local sequence presented on a regular Go board."""

    key: str
    category: str
    title: str
    level: str
    board_size: int
    intro: str
    objective: str
    initial_stones: tuple[tuple[int, Point], ...]
    moves: tuple[TrainingMove, ...]
    conclusion: str


def _rectangle_perimeter(
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> set[Point]:
    points = {(top, col) for col in range(left, right + 1)}
    points.update((bottom, col) for col in range(left, right + 1))
    points.update((row, left) for row in range(top + 1, bottom))
    points.update((row, right) for row in range(top + 1, bottom))
    return points


def _surrounded_eye_stones(
    black_frame: tuple[int, int, int, int],
    white_frame: tuple[int, int, int, int],
    black_fill: Iterable[Point] = (),
) -> tuple[tuple[int, Point], ...]:
    black_points = _rectangle_perimeter(*black_frame) | set(black_fill)
    white_points = _rectangle_perimeter(*white_frame)
    return tuple(
        [(BLACK, point) for point in sorted(black_points)]
        + [(WHITE, point) for point in sorted(white_points)]
    )


_STRAIGHT_THREE = _surrounded_eye_stones(
    black_frame=(2, 2, 4, 6),
    white_frame=(1, 1, 5, 7),
)

_BENT_THREE = _surrounded_eye_stones(
    black_frame=(2, 2, 5, 5),
    white_frame=(1, 1, 6, 6),
    black_fill=((4, 4),),
)

_BULKY_FIVE = _surrounded_eye_stones(
    black_frame=(2, 2, 6, 6),
    white_frame=(1, 1, 7, 7),
    black_fill=((3, 3), (3, 5), (5, 3), (5, 5)),
)


TRAINING_LESSONS: tuple[TrainingLesson, ...] = (
    TrainingLesson(
        key="hoshi_33_invasion",
        category=JOSEKI_CATEGORY,
        title="星位：三三侵入基础型",
        level="入门 · 12 手",
        board_size=19,
        intro=(
            "黑棋先占左上星位，白棋直接侵入三三。这个基础型展示最典型的利益交换："
            "白方取得角地，黑方获得面向边和中央的外势。"
        ),
        objective="逐手找出侵入方安定、守角方筑墙并补住断点的次序。",
        initial_stones=((BLACK, (3, 3)),),
        moves=(
            TrainingMove(
                WHITE,
                (2, 2),
                "侵入三三",
                "白方想立即在角里取得根据，应落在哪里？",
                "找左上角第三线与第三线的交点。",
                "三三离两条棋盘边都很近，做眼所需空间少；代价是主动让黑方在外面筑势。",
            ),
            TrainingMove(
                BLACK,
                (2, 3),
                "选择挡的方向",
                "黑方先限制白棋往哪一边发展？",
                "在侵入子右侧紧贴落子。",
                "挡住右侧，把白棋赶向左边；实战中也可从另一边挡，方向应由外围子力决定。",
            ),
            TrainingMove(
                WHITE,
                (3, 2),
                "向边上长",
                "白方怎样扩大眼位并保持连接？",
                "沿左边第三线向下长一手。",
                "贴着边长能增加气和角内空间，也避免棋子被黑棋分断。",
            ),
            TrainingMove(
                BLACK,
                (4, 3),
                "继续挡住",
                "黑方怎样一边压缩角地、一边形成外壁？",
                "在白棋新子的右侧落子。",
                "紧贴挡住，迫使白棋在低位爬；黑棋的棋子则朝中央连成墙。",
            ),
            TrainingMove(
                WHITE,
                (4, 2),
                "低位爬",
                "白方要保持棋块连贯，应怎样走？",
                "继续沿左边第三线向下。",
                "爬虽然位置低，却能稳定增加角内空间；此处先求活，再谈外围利益。",
            ),
            TrainingMove(
                BLACK,
                (5, 3),
                "压低白棋",
                "黑方怎样延续外势方向？",
                "仍走在白棋右侧。",
                "连续挡让白棋局限在边角，同时保持黑墙没有断点。",
            ),
            TrainingMove(
                WHITE,
                (5, 2),
                "再长一手",
                "白方如何继续确保足够的做眼空间？",
                "沿原方向再爬一格。",
                "这手把角部空间拉长，并为随后从另一边扳回留下余地。",
            ),
            TrainingMove(
                BLACK,
                (6, 3),
                "外壁伸长",
                "黑方怎样避免墙上留下可切断的毛病？",
                "在黑墙末端继续伸长。",
                "伸长既保持对白棋的封锁，也补强墙的连接；若停得过早，白方可能利用断点反击。",
            ),
            TrainingMove(
                WHITE,
                (1, 2),
                "二路扳",
                "白方怎样回到角上，准备做出完整眼形？",
                "在最初三三子的正上方扳。",
                "从上方扳能扩大角地并逼黑棋应对，是把低位长出的棋与角部眼形连起来的关键。",
            ),
            TrainingMove(
                BLACK,
                (1, 3),
                "挡住扳头",
                "黑方怎样阻止白棋向上边扩大？",
                "紧贴白9右侧挡住。",
                "挡住可限制白方边上发展，同时让黑棋外壁保持连续。",
            ),
            TrainingMove(
                WHITE,
                (1, 1),
                "角上长",
                "白方最后怎样把角部做活形整理清楚？",
                "沿二路向角上长。",
                "向角上长后，白棋拥有足够的眼位与气，角部基本安定。",
            ),
            TrainingMove(
                BLACK,
                (2, 5),
                "补住外壁断点",
                "黑方筑墙后，最后要照顾什么薄弱处？",
                "在星位右侧隔一路补强。",
                "这手守住外壁的切断弱点，使黑方能够真正利用面向中央和上边的厚势。",
            ),
        ),
        conclusion=(
            "结果要点：白方以低位换到角地，黑方以角地换到外势。黑方从哪边挡、外势是否有价值，"
            "必须结合全盘判断。"
        ),
    ),
    TrainingLesson(
        key="hoshi_low_approach",
        category=JOSEKI_CATEGORY,
        title="星位：低挂简明参考型",
        level="入门 · 5 手",
        board_size=19,
        intro=(
            "白方从上边低挂左上星位，双方分别照顾两条边。本课采用便于理解方向的短型，"
            "重点是挂角、安定和拆边之间的次序。"
        ),
        objective="理解双方为何先取得根据，再把棋子向宽阔一侧展开。",
        initial_stones=((BLACK, (3, 3)),),
        moves=(
            TrainingMove(
                WHITE,
                (2, 5),
                "低挂星位",
                "白方想试探黑角，同时在上边发展，应走在哪里？",
                "在星位右上方的小飞位置。",
                "低挂兼顾进角和在上边安定，迫使黑方决定守角还是从另一侧发展。",
            ),
            TrainingMove(
                BLACK,
                (5, 2),
                "向左边展开",
                "黑方选择兼顾左边，应怎样落子？",
                "从星位向左边作低位展开。",
                "这手先在左边取得根据，也让原星位棋子不必孤军应战。",
            ),
            TrainingMove(
                WHITE,
                (1, 3),
                "二路滑入",
                "黑方转向左边后，白方怎样利用角部空间？",
                "走到星位正上方的二路。",
                "二路滑入以较低位置换取角上实利，并与低挂子相互照应。",
            ),
            TrainingMove(
                BLACK,
                (2, 2),
                "守住角部",
                "黑方怎样限制白棋继续深入角里？",
                "在滑入子的左下方挡住。",
                "挡住后黑角仍有明确根据，同时把白棋的发展方向引向上边。",
            ),
            TrainingMove(
                WHITE,
                (2, 8),
                "上边拆开",
                "白方最后怎样让低挂子获得舒展空间？",
                "沿上边向右作宽拆。",
                "拆边给白棋留下做眼和发展空间，避免几颗棋挤在角上形成重复。",
            ),
        ),
        conclusion=(
            "结果要点：黑方稳住左边与角部，白方在上边得到安定。这个短型是方向训练示例；"
            "现代实战还有大量夹击、靠压与脱先分支。"
        ),
    ),
    TrainingLesson(
        key="straight_three_live",
        category=LIFE_AND_DEATH_CATEGORY,
        title="直三：黑先做活",
        level="基础 · 1 手",
        board_size=9,
        intro=(
            "黑棋外围已经被白棋包围，内部只剩横向三个空点。现在轮黑方，目标是净活。"
        ),
        objective="找到能把一个大眼分成两个真眼的唯一要点。",
        initial_stones=_STRAIGHT_THREE,
        moves=(
            TrainingMove(
                BLACK,
                (3, 4),
                "占据直三中心",
                "黑方应占哪一点，才能立即做出两只眼？",
                "三个空点里，中间点最重要。",
                "中心一子把左右两个空点完全分开；白棋无法同时填掉两只真眼，因此黑棋净活。",
            ),
        ),
        conclusion="核心口诀：直三中间是要点——我方先占可做活，对方先占可点杀。",
    ),
    TrainingLesson(
        key="straight_three_kill",
        category=LIFE_AND_DEATH_CATEGORY,
        title="直三：白先点杀",
        level="基础 · 3 手",
        board_size=9,
        intro=(
            "仍是被完全包围的直三，但这次轮白方。白棋要阻止黑棋把内部空间分成两眼。"
        ),
        objective="先抢生死要点，再看清黑方收气时为什么仍无法逃脱。",
        initial_stones=_STRAIGHT_THREE,
        moves=(
            TrainingMove(
                WHITE,
                (3, 4),
                "中心点杀",
                "白方第一手应破掉哪个要点？",
                "敌之要点也是我之要点：抢占正中。",
                "白棋占住中心后，黑棋不能再把左右分成两眼；白子也以两端空点为气，暂时提不掉。",
            ),
            TrainingMove(
                BLACK,
                (3, 3),
                "从一端收气",
                "黑方若想提掉白子，会先从哪里收气？",
                "左右对称，本题选择左端。",
                "黑棋填左端后，白棋中心子还剩右端一气；同时整块黑棋也只剩右端这一口气。",
            ),
            TrainingMove(
                WHITE,
                (3, 5),
                "占最后一气",
                "白方怎样完成杀棋？",
                "落在右端的共同最后一气。",
                "白棋抢先填掉黑棋最后一气并提走整块黑棋；这说明先占直三中心后，攻击方在收气次序上领先。",
            ),
        ),
        conclusion="核心方法：先占中心制造内气优势，再读清双方最后一气，而不是只看眼区大小。",
    ),
    TrainingLesson(
        key="bent_three_live",
        category=LIFE_AND_DEATH_CATEGORY,
        title="曲三：黑先做活",
        level="基础 · 1 手",
        board_size=9,
        intro=(
            "黑棋内部有三个呈直角相连的空点，称为曲三。黑方只有一手能把它们分成独立眼位。"
        ),
        objective="识别曲三的转折点，并理解“连接最多的点”为什么是要点。",
        initial_stones=_BENT_THREE,
        moves=(
            TrainingMove(
                BLACK,
                (3, 3),
                "占曲三拐点",
                "黑方应走在哪个转折处？",
                "找同时连接另外两个空点的拐角。",
                "拐点一子把右侧与下侧两个空点隔开，各自成为真眼；若走在末端，剩余空点仍连成一片。",
            ),
        ),
        conclusion="核心口诀：曲三拐点是要点。判断时可先找眼区中邻接空点最多的位置。",
    ),
    TrainingLesson(
        key="bulky_five_kill",
        category=LIFE_AND_DEATH_CATEGORY,
        title="梅花五：白先点杀",
        level="基础 · 1 手",
        board_size=9,
        intro=(
            "黑棋看似围出五个空点，但它们呈十字形相连，尚未形成两只独立真眼。白方先行。"
        ),
        objective="找到梅花五的核心要点，破坏黑方分眼的可能。",
        initial_stones=_BULKY_FIVE,
        moves=(
            TrainingMove(
                WHITE,
                (4, 4),
                "中心一击",
                "白方应先占十字形的哪一点？",
                "找与上下左右四个空点都相邻的中心。",
                "中心是黑方分眼效率最高的点。白棋先占后，四个末端无法被黑棋用一手组织成两只安全的眼，形成梅花五点杀。",
            ),
        ),
        conclusion="核心方法：大眼先找“枢纽点”。连接分支最多的中心，往往就是攻守双方共同的要点。",
    ),
)


def lessons_for_category(category: str) -> tuple[TrainingLesson, ...]:
    """Return lessons in their curated display order."""

    return tuple(lesson for lesson in TRAINING_LESSONS if lesson.category == category)


def lesson_by_title(category: str, title: str) -> TrainingLesson:
    """Look up a lesson selected by the two GUI comboboxes."""

    for lesson in TRAINING_LESSONS:
        if lesson.category == category and lesson.title == title:
            return lesson
    raise KeyError((category, title))


def _neighbors(size: int, point: Point) -> tuple[Point, ...]:
    row, col = point
    result: list[Point] = []
    if row > 0:
        result.append((row - 1, col))
    if row + 1 < size:
        result.append((row + 1, col))
    if col > 0:
        result.append((row, col - 1))
    if col + 1 < size:
        result.append((row, col + 1))
    return tuple(result)


def _group_and_liberties(
    board: list[list[int]],
    start: Point,
) -> tuple[set[Point], set[Point]]:
    size = len(board)
    color = board[start[0]][start[1]]
    group: set[Point] = set()
    liberties: set[Point] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        if point in group:
            continue
        group.add(point)
        for neighbor in _neighbors(size, point):
            value = board[neighbor[0]][neighbor[1]]
            if value == EMPTY:
                liberties.add(neighbor)
            elif value == color and neighbor not in group:
                pending.append(neighbor)
    return group, liberties


def _play_training_move(
    board: list[list[int]],
    move: TrainingMove,
) -> None:
    row, col = move.point
    size = len(board)
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"训练落点越界：{move.point}")
    if board[row][col] != EMPTY:
        raise ValueError(f"训练落点已有棋子：{move.point}")

    board[row][col] = move.color
    for neighbor in _neighbors(size, move.point):
        if board[neighbor[0]][neighbor[1]] != opponent(move.color):
            continue
        group, liberties = _group_and_liberties(board, neighbor)
        if not liberties:
            for captured_row, captured_col in group:
                board[captured_row][captured_col] = EMPTY

    own_group, own_liberties = _group_and_liberties(board, move.point)
    if not own_liberties:
        for own_row, own_col in own_group:
            board[own_row][own_col] = EMPTY
        raise ValueError(f"训练次序包含自杀落子：{move.point}")


def position_after(lesson: TrainingLesson, move_count: int) -> BoardHash:
    """Replay the first ``move_count`` moves and return an immutable board."""

    if not 0 <= move_count <= len(lesson.moves):
        raise ValueError("move_count 超出训练次序范围")

    board = [[EMPTY for _ in range(lesson.board_size)] for _ in range(lesson.board_size)]
    for color, point in lesson.initial_stones:
        row, col = point
        if color not in (BLACK, WHITE):
            raise ValueError(f"无效的初始棋子颜色：{color}")
        if not (0 <= row < lesson.board_size and 0 <= col < lesson.board_size):
            raise ValueError(f"初始棋子越界：{point}")
        if board[row][col] != EMPTY:
            raise ValueError(f"初始棋子重叠：{point}")
        board[row][col] = color

    for move in lesson.moves[:move_count]:
        _play_training_move(board, move)
    return tuple(tuple(row) for row in board)
