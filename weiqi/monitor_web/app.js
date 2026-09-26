"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const phases = {
    initializing: "初始化", startup: "启动中", ready: "准备就绪", selfplay: "自我对弈与对手采样",
    training: "模型训练", position_diagnostics: "固定局面诊断", opponent_pool: "固定对手池评估",
    screening: "早期筛选", evaluation: "晋级评估", checkpoint: "保存检查点",
    benchmark: "性能测试", completed: "训练结束", idle: "等待训练", unknown: "等待阶段信息",
  };
  const states = { running: "训练中", paused: "已暂停", completed: "已完成", stopped: "已停止", failed: "训练异常", stale: "数据未更新", historical: "历史记录", empty: "暂无数据" };
  const roles = { training: "训练采样", pool: "对手池", screening: "早期筛选", evaluation: "晋级评估" };
  const eventNames = {
    started: "训练启动", ready: "模型就绪", iteration_started: "开始新一轮", selfplay_schedule: "对弈计划已生成",
    game_started: "对局开始", game_completed: "对局完成", game_finished: "对局完成", training_step: "模型参数更新",
    iteration_completed: "本轮训练完成", evaluation_completed: "评估完成", completed: "训练完成",
    stopped: "训练已停止", failed: "训练发生异常", error: "训练发生异常", paused: "训练已暂停", resumed: "训练已恢复",
    pause: "训练已暂停", resume: "训练已恢复", amp_overflow: "调整混合精度更新",
    position_evaluated: "局面诊断完成", position_completed: "局面诊断完成", benchmark_completed: "性能测试完成",
    heartbeat: "训练心跳", pool_started: "对手池评估开始", pool_completed: "对手池评估完成",
    run_started: "训练启动", run_completed: "训练完成", run_stopped: "训练已停止", run_failed: "训练发生异常",
    game: "对局完成", games_started: "开始批次对弈", search_progress: "对局搜索进行中",
  };
  const numberFormat = new Intl.NumberFormat("zh-CN");
  const urlRun = new URLSearchParams(window.location.search).get("run");
  let selectedRun = urlRun || "";
  let timer = null;
  let controller = null;
  let generation = 0;
  let lastSnapshot = null;
  let runsSignature = "";
  let opponentsSignature = "";
  let eventsSignature = "";
  let chartSignature = "";

  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function count(value) { return finite(value) ? numberFormat.format(value) : "—"; }
  function percent(value) { return finite(value) ? `${(value * 100).toFixed(1)}%` : "—"; }
  function rate(value) { return finite(value) && value >= 0 && value <= 1 ? value : null; }
  function text(id, value) { $(id).textContent = value == null || value === "" ? "—" : String(value); }
  function list(value) { return Array.isArray(value) ? value : []; }
  function phase(value) { return phases[value] || (value ? String(value) : "等待阶段信息"); }
  function date(value) {
    if (value == null || value === "") return null;
    const parsed = new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  function timestamp(value, includeDate = false) {
    const parsed = date(value);
    if (!parsed) return "—";
    return parsed.toLocaleString("zh-CN", includeDate
      ? { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }
      : { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }
  function age(seconds) {
    if (!finite(seconds)) return "更新时间未知";
    if (seconds < 5) return "刚刚更新";
    if (seconds < 60) return `${Math.floor(seconds)} 秒前更新`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前更新`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前更新`;
    return `${Math.floor(seconds / 86400)} 天前更新`;
  }
  function opponentName(value) {
    if (!value) return "对手未记录";
    if (value === "candidate_self") return "候选模型自我对弈";
    if (value === "candidate") return "候选模型";
    if (value === "best") return "当前最佳模型";
    const accepted = /^accepted_(\d+)$/.exec(value);
    const milestone = /^milestone_(\d+)$/.exec(value);
    if (accepted) return Number(accepted[1]) === 0 ? "初始最佳模型" : `第 ${Number(accepted[1])} 轮最佳模型`;
    if (milestone) return `第 ${Number(milestone[1])} 轮存档模型`;
    return String(value);
  }
  function shortHash(value) { return value ? String(value).slice(0, 12) : "未记录"; }
  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = String(value);
    return node;
  }
  function svgElement(tag, attrs = {}, value) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, String(val));
    if (value !== undefined) node.textContent = String(value);
    return node;
  }
  function metricValue(id, value, unit) {
    const node = $(id);
    node.replaceChildren(document.createTextNode(value));
    if (unit) node.append(element("span", "value-unit", unit));
  }
  function setConnection(online) {
    $("connection").className = `connection ${online ? "online" : "offline"}`;
    text("connection-text", online ? "服务已连接" : "连接中断");
  }
  function errorMessage(error) {
    if (error && error.name === "TimeoutError") return "读取训练数据超时，正在自动重试。";
    if (error instanceof TypeError) return "无法连接监控服务，正在自动重试。";
    return error?.message || "读取训练数据失败，正在自动重试。";
  }
  async function fetchJSON(path, signal) {
    const response = await fetch(path, { signal, cache: "no-store", headers: { Accept: "application/json" } });
    let data;
    try { data = await response.json(); } catch { throw new Error(`服务返回了无法读取的数据（HTTP ${response.status}）。`); }
    if (!response.ok || data.error) throw new Error(data.error || `读取失败（HTTP ${response.status}）`);
    return data;
  }
  function renderRuns(data) {
    const runs = list(data.runs);
    const oldSelected = selectedRun;
    if (!runs.some((run) => run.id === selectedRun)) selectedRun = runs.some((run) => run.id === data.default_run) ? data.default_run : runs[0]?.id || "";
    const signature = JSON.stringify(runs.map((run) => [run.id, run.name]));
    if (signature !== runsSignature || $("run-select").disabled) {
      runsSignature = signature;
      const options = runs.map((run) => {
        const option = element("option", "", run.name || run.id);
        option.value = run.id;
        return option;
      });
      if (!options.length) options.push(element("option", "", "暂无训练记录"));
      $("run-select").replaceChildren(...options);
    }
    $("run-select").disabled = !runs.length;
    $("run-select").value = selectedRun;
    if (oldSelected !== selectedRun) resetRun();
    return runs;
  }
  function resetRun() {
    lastSnapshot = null;
    opponentsSignature = eventsSignature = chartSignature = "";
    $("main").setAttribute("aria-busy", "true");
    $("dashboard").style.opacity = ".55";
    text("page-description", selectedRun ? "正在读取所选训练记录…" : "读取训练记录，追踪候选模型与对手的表现。");
    $("warnings").replaceChildren();
  }
  function currentBatch(snapshot) {
    const current = snapshot.current || {};
    const activePhase = snapshot.status?.phase;
    if (current.match && current.match.opponent !== "candidate_self") return { ...current.match, historical: !["running", "paused"].includes(snapshot.status?.state) };
    // Self-play never borrows an older evaluation as a current candidate win rate.
    if (activePhase === "selfplay") return null;
    const latest = snapshot.latest || {};
    const history = list(snapshot.history).slice().sort((a, b) => (b.iteration || 0) - (a.iteration || 0));
    for (const key of ["evaluation", "screening"]) {
      const row = history.find((item) => item[key] && (finite(item[key].win_rate) || item[key].games > 0));
      if (row) return { ...row[key], phase: key, iteration: row.iteration, historical: true };
      if (latest[key] && (finite(latest[key].win_rate) || latest[key].games > 0)) {
        const metric = latest[key];
        return { ...metric, win_rate: finite(metric.win_rate) ? metric.win_rate : finite(metric.wins) && metric.games > 0 && !metric.unknown ? metric.wins / metric.games : null, phase: key, iteration: latest.iteration, historical: true };
      }
    }
    return null;
  }
  function renderMetrics(snapshot) {
    const current = snapshot.current || {};
    const status = snapshot.status || {};
    const batch = currentBatch(snapshot);
    const winRate = rate(batch?.win_rate);
    const scoreRate = rate(batch?.score_rate);
    text("rate-title", batch?.historical ? `最近${phase(batch.phase)}胜率` : "候选模型胜率");
    metricValue("win-rate", winRate == null ? "—" : (winRate * 100).toFixed(1), winRate == null ? "" : "%");
    $("win-rate-track").style.width = `${(winRate || 0) * 100}%`;
    const batchGames = batch?.games;
    text("rate-caption", batch
      ? `${batch.historical ? `第 ${count(batch.iteration)} 轮 · ` : ""}${status.phase === "selfplay" ? `本批次对手对局 · ${opponentName(batch.opponent || batch.opponent_name || batch.name)}` : phase(batch.phase || status.phase)} · ${batchGames === 0 ? "等待首局结果" : `${count(batch.wins)} 胜 / ${count(batchGames)} 局`}`
      : status.phase === "selfplay" ? "纯自我对弈不产生候选胜率" : "尚无可用的候选模型评估");
    text("score-rate", `含和棋折算得分率 ${percent(scoreRate)}`);
    const active = list(current.active_games).filter((game) => game.opponent_name);
    const activeOpponents = active.map((game) => ({ name: game.opponent_name, sha256: game.opponent_sha256 }));
    const scheduled = list(current.opponents);
    const historicalOpponent = !activeOpponents.length && !current.match && !scheduled.length && batch?.historical;
    const sources = activeOpponents.length ? activeOpponents : current.match?.opponent ? [{ name: current.match.opponent, sha256: current.match.opponent_sha256 }] : scheduled.length ? scheduled : historicalOpponent ? [{ name: batch.opponent || batch.opponent_name || batch.name, sha256: batch.opponent_sha256 || batch.sha256 }] : [];
    text("opponent-title", historicalOpponent ? "最近评测对手" : "当前对手");
    const unique = Array.from(new Map(sources.map((opponent) => [`${opponent.name}:${opponent.sha256 || ""}`, opponent])).values());
    const pureSelf = unique.length === 1 && unique[0].name === "candidate_self";
    if (unique.length) {
      text("current-opponent", unique.length > 1 ? `${unique.length} 个${activeOpponents.length ? "进行中" : "计划"}对手` : opponentName(unique[0].name));
      $("current-opponent").title = unique.map((opponent) => opponentName(opponent.name)).join("、");
      text("opponent-caption", unique.length > 1 ? unique.map((opponent) => opponentName(opponent.name)).join(" / ") : pureSelf ? "同一候选模型执黑、执白" : activeOpponents.length ? `${active.length} 局正在进行` : historicalOpponent ? `第 ${count(batch.iteration)} 轮 · ${phase(batch.phase)}` : status.state === "running" ? "当前阶段记录的对手" : "最近阶段记录的对手");
      text("opponent-hash", unique.length > 1 ? "对手权重详见下方记录" : `SHA256 ${shortHash(unique[0].sha256)}`);
      $("opponent-hash").title = unique.length === 1 ? unique[0].sha256 || "权重标识未记录" : "";
    } else {
      text("current-opponent", status.phase === "training" ? "模型参数训练中" : "暂无进行中的对局");
      text("opponent-caption", "最近的评估对手可在下方记录查看");
      text("opponent-hash", "权重标识 —");
      $("current-opponent").removeAttribute("title");
      $("opponent-hash").removeAttribute("title");
    }
    metricValue("iteration", count(current.iteration), finite(current.iteration) ? "轮" : "");
    text("iteration-caption", `已完成 ${count(current.completed_iteration)} 轮`);
    text("best-iteration", finite(current.best_iteration) ? current.best_iteration === 0 ? "当前最佳为初始模型" : `当前最佳来自第 ${count(current.best_iteration)} 轮` : "当前最佳轮次未记录");
    const progress = current.progress;
    if (progress && finite(progress.completed)) {
      metricValue("progress-value", count(progress.completed), finite(progress.total) ? ` / ${count(progress.total)}` : "");
      const ratio = progress.total > 0 ? Math.min(100, Math.max(0, progress.completed / progress.total * 100)) : 0;
      $("progress-fill").style.width = `${ratio}%`;
      if (progress.total > 0) $("phase-progress").setAttribute("aria-valuenow", String(Math.round(ratio)));
      else $("phase-progress").removeAttribute("aria-valuenow");
      const unit = { games: "局对弈", steps: "次参数更新", positions: "个诊断局面" }[progress.unit] || "项任务";
      text("progress-caption", `已完成 ${count(progress.completed)} ${unit}`);
    } else {
      metricValue("progress-value", "—", "");
      $("progress-fill").style.width = "0%";
      $("phase-progress").removeAttribute("aria-valuenow");
      text("progress-caption", "当前阶段暂无可量化进度");
    }
    text("training-steps", `累计训练步数 ${count(current.training_steps)}`);
    text("batch-tag", batch ? batch.historical ? "最近完成" : "当前批次" : status.phase === "selfplay" ? "自我对弈" : "等待评估");
    text("detail-title", batch?.historical ? "最近评估表现" : "当前对局表现");
    text("batch-context", batch ? `${batch.historical ? `第 ${count(batch.iteration)} 轮 · ` : ""}${phase(batch.phase || status.phase)} · 对手：${opponentName(batch.opponent || batch.opponent_name || batch.name)}` : status.phase === "selfplay" ? "纯自我对弈的黑白胜负不代表候选模型胜率。" : "候选模型的对手对局会显示在这里。");
    const total = batch?.games > 0 ? batch.games : 0;
    for (const key of ["wins", "losses", "draws", "truncated"]) {
      text(key, count(batch?.[key]));
      $(`bar-${key}`).style.width = `${total && finite(batch?.[key]) ? Math.max(0, Math.min(100, batch[key] / total * 100)) : 0}%`;
    }
    $("bar-unknown").style.width = `${total && batch?.unknown ? Math.max(0, Math.min(100, batch.unknown / total * 100)) : 0}%`;
    $("result-bar").setAttribute("aria-label", batch ? `${count(batch.wins)} 胜，${count(batch.losses)} 负，${count(batch.draws)} 和棋，${count(batch.truncated)} 截断，${count(batch.unknown || 0)} 结果未知` : "暂无候选模型对局结果");
    $("unknown-results").hidden = !(batch?.unknown > 0);
    text("unknown-results", `${count(batch?.unknown)} 局旧记录缺少候选方结果，无法计算完整胜率。`);
    const paired = batch?.paired;
    $("confidence-note").hidden = !(finite(paired?.lower) && finite(paired?.upper));
    text("confidence-note", paired ? `成对得分 ${finite(paired.confidence) ? `${Math.round(paired.confidence * 100)}% ` : ""}置信区间 ${percent(paired.lower)}–${percent(paired.upper)} · ${count(paired.complete_pairs)} 对完整对局` : "");
    text("device", [current.device, current.gpu].filter(Boolean).join(" · ") || "未记录");
    text("replay-samples", finite(current.replay_samples) ? `${count(current.replay_samples)} 条` : "—");
    const loss = finite(current.loss) ? current.loss : current.loss?.loss ?? current.loss?.total_loss ?? current.loss?.total;
    text("training-loss", finite(loss) ? loss.toFixed(4) : "—");
  }
  function renderChart(history) {
    const rows = list(history).filter((row) => finite(row.iteration)).slice().sort((a, b) => a.iteration - b.iteration);
    const signature = JSON.stringify(rows.map((row) => [row.iteration, row.evaluation, row.screening, row.promoted]));
    text("history-count", `${rows.length} 轮记录`);
    if (signature === chartSignature) return;
    chartSignature = signature;
    const svg = $("trend-chart");
    const children = [svgElement("title", { id: "chart-title" }, "候选模型评估胜率趋势")];
    const available = rows.some((row) => rate(row.evaluation?.win_rate) != null || rate(row.screening?.win_rate) != null);
    children.push(svgElement("desc", { id: "chart-description" }, available ? "横轴为训练轮次，纵轴为胜场除以总对局。绿色表示晋级评估，黄色表示早期筛选。缺失数据或对手权重变化时断开连线。聚焦数据点可读取胜率和对手。" : "尚无候选模型评估数据。纯自我对弈不产生候选胜率。"));
    const box = { left: 48, right: 735, top: 20, bottom: 213 };
    const min = rows.length ? rows[0].iteration : 0;
    const max = rows.length ? rows[rows.length - 1].iteration : 4;
    const x = (value) => max === min ? (box.left + box.right) / 2 : box.left + ((value - min) / (max - min)) * (box.right - box.left);
    const y = (value) => box.bottom - value * (box.bottom - box.top);
    for (const value of [0, .25, .5, .75, 1]) {
      children.push(svgElement("line", { x1: box.left, x2: box.right, y1: y(value), y2: y(value), class: "chart-grid" }));
      children.push(svgElement("text", { x: box.left - 12, y: y(value) + 3, "text-anchor": "end", class: "chart-axis-label" }, `${Math.round(value * 100)}%`));
    }
    const ticks = rows.length ? rows.filter((_, index) => index % Math.max(1, Math.ceil(rows.length / 7)) === 0 || index === rows.length - 1) : [];
    for (const row of ticks) children.push(svgElement("text", { x: x(row.iteration), y: box.bottom + 22, "text-anchor": "middle", class: "chart-axis-label" }, row.iteration));
    children.push(svgElement("text", { x: box.right, y: 251, "text-anchor": "end", class: "chart-x-title" }, "训练轮次"));
    for (const [key, color, dash] of [["screening", "#c3a36b", "5 5"], ["evaluation", "#2d7b65", ""]]) {
      let path = "";
      let previous = null;
      const points = [];
      for (const row of rows) {
        const metric = row[key];
        const value = rate(metric?.win_rate);
        if (value == null) { previous = null; continue; }
        const identity = metric.opponent_sha256 || metric.sha256 || "";
        const budget = metric.simulations_per_move;
        const continuous = identity && finite(budget) && previous && previous.iteration + 1 === row.iteration && previous.identity === identity && previous.budget === budget;
        path += `${continuous ? "L" : "M"}${x(row.iteration).toFixed(2)},${y(value).toFixed(2)} `;
        previous = { iteration: row.iteration, identity, budget };
        const label = `第 ${row.iteration} 轮，${roles[key]}，胜率 ${percent(value)}，${count(metric.games)} 局，每手搜索 ${count(budget)} 次，对手 ${opponentName(metric.opponent || metric.opponent_name || metric.name)}，权重 ${shortHash(metric.opponent_sha256 || metric.sha256)}`;
        const circle = svgElement("circle", { cx: x(row.iteration), cy: y(value), r: 4.5, fill: color, class: "chart-point", tabindex: "0", "aria-label": label });
        circle.append(svgElement("title", {}, label));
        points.push(circle);
      }
      children.push(svgElement("path", { d: path, stroke: color, "stroke-dasharray": dash, class: "chart-path" }));
      children.push(...points);
    }
    svg.replaceChildren(...children);
    $("chart-empty").hidden = available;
  }
  function renderOpponents(opponents) {
    const rows = list(opponents);
    const signature = JSON.stringify(rows);
    if (signature === opponentsSignature) return;
    opponentsSignature = signature;
    const body = $("opponents-body");
    if (!rows.length) {
      const row = element("tr");
      const cell = element("td", "table-empty", "尚无对手记录；对局开始后会自动更新");
      cell.colSpan = 8;
      row.append(cell);
      body.replaceChildren(row);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const opponent of rows) {
      const row = element("tr");
      const cell = element("td");
      const wrapper = element("div", "opponent-cell");
      const icon = element("span", "table-stone");
      icon.setAttribute("aria-hidden", "true");
      const description = element("div");
      const name = element("strong", "", opponentName(opponent.name || opponent.opponent));
      name.title = String(opponent.name || opponent.opponent || "对手未记录");
      const hash = element("small", "mono", shortHash(opponent.sha256 || opponent.opponent_sha256));
      hash.title = String(opponent.sha256 || opponent.opponent_sha256 || "权重标识未记录");
      description.append(name, hash);
      wrapper.append(icon, description);
      cell.append(wrapper);
      row.append(cell);
      const roleCell = element("td");
      roleCell.append(element("span", `role-tag ${roles[opponent.role] ? opponent.role : ""}`, roles[opponent.role] || "其他"));
      row.append(roleCell, element("td", "", count(opponent.iteration)), element("td", "", count(opponent.games)));
      const isSelf = opponent.name === "candidate_self" || opponent.opponent === "candidate_self";
      const results = isSelf ? "不适用" : `${count(opponent.wins)} / ${count(opponent.losses)} / ${count(opponent.draws)}`;
      row.append(element("td", "", results), element("td", "", count(opponent.truncated)));
      const winCell = element("td");
      const winRate = isSelf ? null : rate(opponent.win_rate);
      const rateWrapper = element("div", "table-rate");
      rateWrapper.append(element("span", "", percent(winRate)));
      if (winRate != null) {
        const track = element("span", "table-rate-track");
        track.setAttribute("aria-hidden", "true");
        const fill = element("span");
        fill.style.width = `${winRate * 100}%`;
        track.append(fill);
        rateWrapper.append(track);
      }
      winCell.append(rateWrapper);
      row.append(winCell, element("td", "", percent(isSelf ? null : rate(opponent.score_rate))));
      fragment.append(row);
    }
    body.replaceChildren(fragment);
  }
  function eventDescription(event) {
    const pieces = [];
    if (finite(event.iteration)) pieces.push(`第 ${event.iteration} 轮`);
    if (event.opponent_name || event.opponent) pieces.push(`对手 ${opponentName(event.opponent_name || event.opponent)}`);
    if (finite(event.index)) pieces.push(`第 ${event.index + 1} 局`);
    if (finite(event.game_index)) pieces.push(`第 ${event.game_index + 1} 局`);
    if (event.candidate_color) pieces.push(`候选执${event.candidate_color === 1 || event.candidate_color === "black" || event.candidate_color === "B" ? "黑" : "白"}`);
    const resultNames = { win: "候选获胜", loss: "候选落败", draw: "和棋", truncated: "长度截断", unknown: "候选结果未知", black_win: "黑方获胜", white_win: "白方获胜" };
    if (event.candidate_result) pieces.push(resultNames[event.candidate_result] || String(event.candidate_result));
    else if (event.result && typeof event.result !== "object") pieces.push(resultNames[event.result] || `结果 ${event.result}`);
    if (event.reason === "length_limit") pieces.push("达到长度上限");
    if (finite(event.moves)) pieces.push(`${event.moves} 手`);
    if (finite(event.step)) pieces.push(`参数更新 ${event.step}${finite(event.total) ? ` / ${event.total}` : ""}`);
    if (finite(event.loss)) pieces.push(`损失 ${event.loss.toFixed(4)}`);
    if (finite(event.champion_games)) pieces.push(`历史最佳对弈 ${event.champion_games} 局`);
    if (finite(event.candidate_self_games)) pieces.push(`自我对弈 ${event.candidate_self_games} 局`);
    if (event.promoted === true) pieces.push("候选模型晋升为最佳模型");
    if (event.promoted === false && event.event === "iteration_completed") pieces.push("本轮未晋级");
    if (event.resumed === true) pieces.push("从检查点恢复");
    if (event.message) pieces.push(String(event.message));
    if (event.error) pieces.push(String(event.error));
    return pieces.join(" · ");
  }
  function renderEvents(events) {
    const rows = list(events).slice(0, 40);
    const signature = JSON.stringify(rows);
    if (signature === eventsSignature) return;
    eventsSignature = signature;
    const fragment = document.createDocumentFragment();
    for (const event of rows) {
      const item = element("li", "event-item");
      const icon = element("span", `event-icon ${event.promoted ? "promoted" : ["failed", "error", "run_failed"].includes(event.event) ? "failed" : ""}`);
      icon.setAttribute("aria-hidden", "true");
      const content = element("div", "event-content");
      const title = element("div", "event-title");
      title.append(element("strong", "", eventNames[event.event] || event.event || "训练活动"));
      if (event.phase) title.append(document.createTextNode(phase(event.phase)));
      content.append(title);
      const details = eventDescription(event);
      if (details) content.append(element("p", "event-description", details));
      const time = element("time", "event-time", timestamp(event.time || event.timestamp));
      const parsed = date(event.time || event.timestamp);
      if (parsed) { time.dateTime = parsed.toISOString(); time.title = timestamp(event.time || event.timestamp, true); }
      item.append(icon, content, time);
      fragment.append(item);
    }
    if (!rows.length) fragment.append(element("li", "event-empty", "暂无训练动态；历史运行可能未记录实时事件"));
    const scroll = $("event-list").scrollTop;
    $("event-list").replaceChildren(fragment);
    $("event-list").scrollTop = scroll;
  }
  function renderSnapshot(snapshot) {
    lastSnapshot = snapshot;
    const status = snapshot.status || {};
    $("main").setAttribute("aria-busy", "false");
    $("dashboard").style.opacity = "";
    $("dashboard").hidden = false;
    $("empty-state").hidden = true;
    text("page-description", snapshot.run?.name ? `正在观测 ${snapshot.run.name}` : "追踪训练进度、候选模型胜率与每轮对手。");
    text("state-badge", states[status.state] || "状态未知");
    $("state-badge").className = `status-badge ${states[status.state] ? status.state : ""}`;
    text("phase-label", phase(status.phase));
    text("status-detail", status.reason || "数据来自本地训练记录");
    $("status-detail").title = status.reason || "";
    text("data-age", age(status.age_seconds));
    $("data-age").title = `训练数据更新时间：${timestamp(status.updated_at, true)}`;
    text("last-refresh", `页面刷新 ${timestamp(snapshot.server_time || Date.now())}`);
    text("footer-time", `训练数据更新于 ${timestamp(status.updated_at, true)}`);
    const warningFragment = document.createDocumentFragment();
    for (const warning of list(snapshot.warnings)) warningFragment.append(element("div", "notice warning-notice", warning));
    $("warnings").replaceChildren(warningFragment);
    renderMetrics(snapshot);
    renderChart(snapshot.history);
    renderOpponents(snapshot.opponents);
    renderEvents(snapshot.events);
  }
  async function refresh() {
    const thisGeneration = generation;
    clearTimeout(timer);
    if (controller) controller.abort();
    controller = new AbortController();
    const requestController = controller;
    const timeout = setTimeout(() => requestController.abort(new DOMException("请求超时", "TimeoutError")), 12000);
    try {
      const runData = await fetchJSON("/api/runs", requestController.signal);
      if (thisGeneration !== generation) return;
      const runs = renderRuns(runData);
      if (runs.length && selectedRun) {
        const requestedRun = selectedRun;
        const snapshot = await fetchJSON(`/api/snapshot?run=${encodeURIComponent(requestedRun)}`, requestController.signal);
        if (thisGeneration !== generation || selectedRun !== requestedRun) return;
        renderSnapshot(snapshot);
      } else {
        $("dashboard").hidden = true;
        $("empty-state").hidden = false;
        $("main").setAttribute("aria-busy", "false");
        text("page-description", "尚未发现训练运行，启动训练后会自动显示。");
        text("last-refresh", `页面刷新 ${timestamp(Date.now())}`);
      }
      setConnection(true);
      $("error-banner").hidden = true;
    } catch (error) {
      if (thisGeneration !== generation || error?.name === "AbortError") return;
      setConnection(false);
      $("main").setAttribute("aria-busy", "false");
      $("dashboard").style.opacity = lastSnapshot ? "" : ".55";
      text("error-message", `${errorMessage(error)}${lastSnapshot ? " 当前保留上次成功读取的数据。" : ""}`);
      $("error-banner").hidden = false;
    } finally {
      clearTimeout(timeout);
      if (thisGeneration === generation) timer = setTimeout(refresh, 2000);
    }
  }
  $("run-select").addEventListener("change", () => {
    selectedRun = $("run-select").value;
    generation += 1;
    if (controller) controller.abort();
    const url = new URL(window.location.href);
    url.searchParams.set("run", selectedRun);
    window.history.replaceState(null, "", url);
    resetRun();
    refresh();
  });
  $("retry-button").addEventListener("click", () => {
    generation += 1;
    if (controller) controller.abort();
    text("connection-text", "重新连接中");
    refresh();
  });
  document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("selected"));
    link.classList.add("selected");
  }));
  window.addEventListener("online", () => { generation += 1; refresh(); });
  window.addEventListener("pagehide", () => { generation += 1; clearTimeout(timer); if (controller) controller.abort(); });
  window.addEventListener("pageshow", (event) => { if (event.persisted) refresh(); });
  refresh();
})();
