const search = document.querySelector('#search');
const position = document.querySelector('#position');
const team = document.querySelector('#team');
const sort = document.querySelector('#sort');
const body = document.querySelector('#rankings');
const count = document.querySelector('#result-count');
const empty = document.querySelector('#empty');

let players = [];

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i += 1; }
      else quoted = !quoted;
    } else if (char === ',' && !quoted) { row.push(field); field = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field);
      if (row.some(value => value !== '')) rows.push(row);
      row = []; field = '';
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [headers, ...records] = rows;
  return records.map(values => Object.fromEntries(headers.map((key, index) => [key, values[index] ?? ''])));
}

function cell(row, value, className = '') {
  const td = document.createElement('td');
  td.textContent = value;
  if (className) td.className = className;
  row.append(td);
  return td;
}

function render() {
  const query = search.value.trim().toLocaleLowerCase();
  const filtered = players.filter(player =>
    (position.value === 'ALL' || player.position === position.value) &&
    (team.value === 'ALL' || player.target_team === team.value) &&
    player.player_name.toLocaleLowerCase().includes(query)
  );
  filtered.sort((a, b) => {
    if (sort.value === 'ppg-asc') return a.projected_ppg - b.projected_ppg;
    if (sort.value === 'name') return a.player_name.localeCompare(b.player_name);
    if (sort.value === 'rank') return a.rank - b.rank || a.position.localeCompare(b.position);
    return b.projected_ppg - a.projected_ppg;
  });
  count.textContent = `${filtered.length} of ${players.length} players`;
  empty.hidden = filtered.length !== 0;
  const fragment = document.createDocumentFragment();
  for (const player of filtered) {
    const row = document.createElement('tr');
    cell(row, String(player.rank).padStart(2, '0'), 'rank');
    cell(row, player.player_name, 'player');
    const positionCell = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `badge ${player.position}`;
    badge.textContent = player.position;
    positionCell.append(badge);
    row.append(positionCell);
    cell(row, player.target_team);
    cell(row, player.projected_ppg.toFixed(2), 'numeric ppg');
    fragment.append(row);
  }
  body.replaceChildren(fragment);
}

async function loadRankings() {
  try {
    const response = await fetch('rankings.csv');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    players = parseCSV(await response.text()).map(player => ({
      ...player,
      rank: Number(player.rank),
      projected_ppg: Number(player.projected_ppg),
    })).filter(player => Number.isFinite(player.rank) && Number.isFinite(player.projected_ppg));
    const teams = [...new Set(players.map(player => player.target_team).filter(Boolean))].sort();
    for (const name of teams) team.add(new Option(name, name));
    render();
  } catch (error) {
    count.textContent = 'Rankings unavailable';
    body.replaceChildren();
    const row = document.createElement('tr');
    cell(row, 'Could not load rankings. Serve the web folder with a local web server and refresh the page.', 'status').colSpan = 5;
    body.append(row);
    console.error(error);
  }
}

for (const control of [search, position, team, sort]) control.addEventListener('input', render);
loadRankings();
