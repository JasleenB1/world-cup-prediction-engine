from pathlib import Path
import urllib.request

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    'results.csv': 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
    'fjelstul_group_standings.csv': 'https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/group_standings.csv',
    'fjelstul_matches.csv': 'https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv',
    'fjelstul_qualified_teams.csv': 'https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/qualified_teams.csv',
    'fjelstul_tournament_standings.csv': 'https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/tournament_standings.csv',
    'fjelstul_host_countries.csv': 'https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/host_countries.csv',
}

for filename, url in FILES.items():
    out = DATA_DIR / filename
    if out.exists():
        print(f'Skipping {filename} - already exists')
        continue
    print(f'Downloading {filename}...')
    urllib.request.urlretrieve(url, out)

print('Done.')
