

import pandas as pd
import numpy as np
import math
from dash import html, dcc, dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import dash_bootstrap_components as dbc
#from jupyter_dash import JupyterDash
from dash import Dash
from plotly import graph_objects as go
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints.playbyplay import PlayByPlay
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

import joblib
import pandas as pd

model = joblib.load('lgb_hyperopt_nba_model.joblib')


# In[59]:


def get_days_by_season(season='2022-23'):
    try:
        gamefinder = LeagueGameFinder(season_nullable=season, league_id_nullable='00')
        games = gamefinder.get_data_frames()[0]
        return (games.GAME_DATE.unique())
    except:
        return (None)


def minutes_to_period_gc(minutes):
    minutes = int(math.floor(minutes / 5))
    gc = 0
    end = 48
    left = end - minutes
    period = None
    overtime = 0
    while left < 0:
        gc = left
        end = end + 5
        if period is None:
            period = 5
        else:
            period = period + 1
        overtime = overtime + 1
        left = end - minutes

    if overtime == 0:
        period = 5
        while left > 0:
            gc = left
            period = period - 1
            minutes = minutes + 12
            left = end - minutes
    return (period, gc)


def gameclock_to_seconds(gameclock, period):
    gameclock = str(gameclock)
    if (gameclock != 'nan') & (gameclock != ''):
        if (gameclock[:2] == 'PT'):
            t = datetime.strptime(gameclock, 'PT%MM%S.%fS')
        else:
            t = datetime.strptime(gameclock, '%M:%S')

        t = int(timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
    else:
        t = 0

    # If overtime, treat like 4th quarter
    if period > 4:
        period = 4

    # add any time from remaining quarters
    return (t + ((4 - period) * 12 * 60))


def reverse_seconds(gameclock, period):
    seconds = gameclock_to_seconds(gameclock, period)
    # Maybe a backwards way of doing this, but for the charts it's better to have the time counting forward
    if period > 4:
        # Overtimes periods are 5 minutes
        return (4 * 12 * 60 + (period - 4) * 5 * 60 - seconds)
    return (4 * 12 * 60 - seconds)


def get_win_prob(away_score, home_score, period, game_clock):
    seconds_left = gameclock_to_seconds(game_clock, period)

    if (period > 4):
        overtime = 1
    else:
        overtime = 0

    home_score = int(home_score)
    away_score = int(away_score)

    if (seconds_left == 0):
        if (away_score > home_score):
            result = 0
        else:
            result = 1
    else:
        predict_row = pd.DataFrame([{'away_score': away_score,
                                     'home_score': home_score,
                                     'remaining_time': seconds_left,
                                     'overtime': overtime,
                                     'margin': home_score - away_score,
                                     'margin_per_sec': (home_score - away_score) / seconds_left
                                     }])

        result = model.predict_proba(predict_row)[0, 1]

    return (result)


# Today's Score Board
def update_live():
    live_game_dict = {}
    for g in scoreboard.ScoreBoard().get_dict()['scoreboard']['games']:
        live_game_dict[g['gameId']] = {k: v for k, v in g.items() if
                                       k in ['period', 'gameClock', 'homeTeam', 'awayTeam', 'periods']}
    global live_update_time
    live_update_time = datetime.today()
    return (live_game_dict)


def get_game_dict(game_date):
    d = datetime.strftime(game_date, '%m/%d/%Y')
    gamefinder = LeagueGameFinder(date_from_nullable=d, date_to_nullable=d, league_id_nullable='00')

    games = gamefinder.get_data_frames()[0]

    game_dict = {}
    for i, g in games.iterrows():
        if ' vs.' in g.MATCHUP:
            team_type = 'home_'
        else:
            team_type = 'away_'

        if g.GAME_ID not in game_dict.keys():
            game_dict[g.GAME_ID] = {}
        game_dict[g.GAME_ID][team_type + 'score'] = g.PTS
        game_dict[g.GAME_ID][team_type + 'name'] = g.TEAM_NAME
        game_dict[g.GAME_ID][team_type + 'id'] = g.TEAM_ABBREVIATION
        game_dict[g.GAME_ID]['minutes'] = g.MIN
        game_dict[g.GAME_ID]['period'], game_dict[g.GAME_ID]['gameClock'] = minutes_to_period_gc(g.MIN)

    # game_dict['prob'] = get_win_prob(game_dict['away_score'], game_dict['home_score'], game_dict['period'], game_dict['game_clock']):

    if d == datetime.strftime(datetime.today(), '%m/%d/%Y'):
        for i, g in game_dict.items():
            if i in live_game_dict.keys():
                game_dict[i] = get_live_game_info(i, live_game_dict)
                game_dict[i]['live'] = True

    return (game_dict)


live_game_dict = update_live()


def get_prob_category(prob_diff):
    if prob_diff >= 0.07: return 'Home Very Positive'
    if prob_diff > 0: return 'Home Positive'
    if prob_diff <= -0.07: return 'Away Very Positive'
    if prob_diff < 0: return 'Away Positive'
    return 'Neutral'

def get_live_game_info(game_id, live_game_dict=None):
    global live_update_time
    if (datetime.today() - live_update_time).seconds > 60:
        live_game_dict = update_live()
    elif live_game_dict is None:
        live_game_dict = update_live()

    v = live_game_dict[game_id]

    game_dict = {'period': v['period'],
                 'game_clock': v['gameClock']
                 }

    for t in ['home', 'away']:
        team_type = t + '_'
        team_key = t + 'Team'

        game_dict[team_type + 'score'] = v[team_key]['score']
        game_dict[team_type + 'name'] = v[team_key]['teamName']
        game_dict[team_type + 'id'] = v[team_key]['teamTricode']

    game_dict['prob'] = get_win_prob(game_dict['away_score'], game_dict['home_score'], game_dict['period'],
                                     game_dict['game_clock'])

    return (game_dict)

def get_live_game_df(game_id, live_game_dict=None):
    global live_update_time
    if (datetime.today() - live_update_time).seconds > 60:
        live_game_dict = update_live()
    elif live_game_dict is None:
        live_game_dict = update_live()

    v = live_game_dict[game_id]
    h_scores = [x['score'] for x in v['homeTeam']['periods']]
    a_scores = [x['score'] for x in v['awayTeam']['periods']]

    periods = [str(x['period']) if x['period'] < 5 else 'OT' + str(x['period'] - 4) for x in v['awayTeam']['periods']]

    df = pd.DataFrame(list(zip(a_scores, h_scores)), columns=['away', 'home'], index=periods)

    home_tot = 0
    away_tot = 0
    prob_tot = np.nan
    old_prob = 0.57

    info_dict = []

    for i, row in df.iterrows():

        period = int(i)
        if (v['gameClock'] == '') | (v['gameClock'] == 'PT00M00.00S'):
            description = 'End of ' + str(period) + ' Period'
        else:
            description = 'Middle of ' + str(period) + ' Period'

        home_tot = home_tot + row['home']
        away_tot = away_tot + row['away']

        if period <= v['period']:
            prob = get_win_prob(away_tot, home_tot, period, v['gameClock'])
            prob_diff = prob - old_prob
            old_prob = prob

        info_dict.append({'away_score': away_tot,
                          'home_score': home_tot,
                          'game_clock': v['gameClock'],
                          'period': period,
                          'prob': prob,
                          'prob_diff': prob_diff,
                          'prob_category': get_prob_category(prob_diff),
                          'description': description}
                         )

    return (pd.DataFrame(info_dict))

def get_game_df(game_id, full=False):
    try:
        pbp_full = PlayByPlay(game_id).get_data_frames()[0]
        pbp_avail = True
    except:
        print("Play By Play not Available Yet!")
        pbp_avail = False
    if len(pbp_full) == 0:
        print("Play By Play not Available Yet!")
        pbp_avail = False

    if (pbp_avail):
        pbp_full.loc[0, 'SCORE'] = '0 - 0'
        if full:
            pbp = pbp_full.loc[pd.isna(pbp_full['SCORE']) == False, :].copy()
        else:
            pbp = pbp_full.loc[((pbp_full['EVENTMSGTYPE'] == 12) & (pbp_full['SCORE'] == '0 - 0')) | (
                        pbp_full['EVENTMSGTYPE'] == 13), :].copy()

        pbp['away_score'] = pbp['SCORE'].apply(lambda x: x.split(' - ')[0])
        pbp['home_score'] = pbp['SCORE'].apply(lambda x: x.split(' - ')[1])
        pbp['description'] = pbp['HOMEDESCRIPTION'].fillna(pbp['VISITORDESCRIPTION']).fillna(pbp['NEUTRALDESCRIPTION'])
        pbp['prob'] = pbp.apply(
            lambda x: get_win_prob(x['away_score'], x['home_score'], x['PERIOD'], x['PCTIMESTRING']), axis=1)
        pbp['prob_diff'] = pbp['prob'] - pbp['prob'].shift()

        pbp['prob_category'] = pbp['prob_diff'].apply(get_prob_category)
    else:
        pbp = get_live_game_df(game_id)

    return (pbp.rename(columns={"PERIOD": "period", "PCTIMESTRING": "game_clock"})[
        ['away_score', 'home_score', 'game_clock', 'period', 'prob', 'prob_diff', 'prob_category', 'description']])

def get_style_data_conditional(selected_rows):
    non_selected_band_color = "rgb(229, 236, 246)"
    selected_band_color = '#98c21f'
    return [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': non_selected_band_color
        },
        {
            'if': {'row_index': 'even'},
            'backgroundColor': "white"
        },
        {
            'if': {'row_index': selected_rows},
            'backgroundColor': selected_band_color,
            'fontWeight': 'bold',
            'color': 'white',
        },
    ]


def blank_fig():
    #         g = pd.DataFrame(
    #         {'period':[1,2,3,4],
    #          'prob':[0.5,0.7,0.3,0]
    #         })
    #         fig = px.bar(g, x="period"
    #                      , y='prob'
    #                     )

    df = pd.DataFrame(columns=['x', 'y'])

    fig = px.scatter(df)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None
    )
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


team_dict = {'ATL': {'full_name': 'Atlanta Hawks', 'color_1': '(225, 68, 52)', 'color_2': '(196, 214, 0)'},
             'BOS': {'full_name': 'Boston Celtics', 'color_1': '(0, 122, 51)', 'color_2': '(139, 111, 78)'},
             'BKN': {'full_name': 'Brooklyn Nets', 'color_1': '(0, 0, 0)', 'color_2': '(255, 255, 255)'},
             'CHA': {'full_name': 'Charlotte Hornets', 'color_1': '(29, 17, 96)', 'color_2': '(0, 120, 140)'},
             'CHI': {'full_name': 'Chicago Bulls', 'color_1': '(206, 17, 65)', 'color_2': '(6, 25, 34)'},
             'CLE': {'full_name': 'Cleveland Cavaliers', 'color_1': '(134, 0, 56)', 'color_2': '(4, 30, 66)'},
             'DAL': {'full_name': 'Dallas Mavericks', 'color_1': '(0, 83, 188)', 'color_2': '(0, 43, 92)'},
             'DEN': {'full_name': 'Denver Nuggets', 'color_1': '(13, 34, 64)', 'color_2': '(255, 198, 39)'},
             'DET': {'full_name': 'Detroit Pistons', 'color_1': '(200, 16, 46)', 'color_2': '(29, 66, 138)'},
             'GSW': {'full_name': 'Golden State Warriors', 'color_1': '(29, 66, 138)', 'color_2': '(255, 199, 44)'},
             'HOU': {'full_name': 'Houston Rockets', 'color_1': '(206, 17, 65)', 'color_2': '(6, 25, 34)'},
             'IND': {'full_name': 'Indiana Pacers', 'color_1': '(0, 45, 98)', 'color_2': '(253, 187, 48)'},
             'LAC': {'full_name': 'Los Angeles Clippers', 'color_1': '(200, 16, 46)', 'color_2': '(29, 66, 148)'},
             'LAL': {'full_name': 'Los Angeles Lakers', 'color_1': '(85, 37, 130)', 'color_2': '(253, 185, 39)'},
             'MEM': {'full_name': 'Memphis Grizzlies', 'color_1': '(93, 118, 169)', 'color_2': '(18, 23, 63)'},
             'MIA': {'full_name': 'Miami Heat', 'color_1': '(152, 0, 46)', 'color_2': '(249, 160, 27)'},
             'MIL': {'full_name': 'Milwaukee Bucks', 'color_1': '(0, 71, 27)', 'color_2': '(240, 235, 210)'},
             'MIN': {'full_name': 'Minnesota Timberwolves', 'color_1': '(12, 35, 64)', 'color_2': '(35, 97, 146)'},
             'NOP': {'full_name': 'New Orleans Pelicans', 'color_1': '(0, 22, 65)', 'color_2': '(225, 58, 62)'},
             'NYK': {'full_name': 'New York Knicks', 'color_1': '(0, 107, 182)', 'color_2': '(245, 132, 38)'},
             'OKC': {'full_name': 'Oklahoma City Thunder', 'color_1': '(0, 125, 195)', 'color_2': '(239, 59, 36)'},
             'ORL': {'full_name': 'Orlando Magic', 'color_1': '(0, 125, 197)', 'color_2': '(196, 206, 211)'},
             'PHI': {'full_name': 'Philadelphia 76ers', 'color_1': '(0, 107, 182)', 'color_2': '(237, 23, 76)'},
             'PHX': {'full_name': 'Phoenix Suns', 'color_1': '(29, 17, 96)', 'color_2': '(229, 95, 32)'},
             'POR': {'full_name': 'Portland Trail Blazers', 'color_1': '(224, 58, 62)', 'color_2': '(6, 25, 34)'},
             'SAC': {'full_name': 'Sacramento Kings', 'color_1': '(91, 43, 130)', 'color_2': '(99, 113, 122)'},
             'SAS': {'full_name': 'San Antonio Spurs', 'color_1': '(196, 206, 211)', 'color_2': '(6, 25, 34)'},
             'TOR': {'full_name': 'Toronto Raptors', 'color_1': '(206, 17, 65)', 'color_2': '(6, 25, 34)'},
             'UTA': {'full_name': 'Utah Jazz', 'color_1': '(0, 43, 92)', 'color_2': '(0, 71, 27)'},
             'WAS': {'full_name': 'Washington Wizards', 'color_1': '(0, 43, 92)', 'color_2': '(227, 24, 55)'}
             }

def get_colors(home_id, away_id):
    home = team_dict[home_id]['color_1']
    away1 = team_dict[away_id]['color_1']
    away2 = team_dict[away_id]['color_2']

    dist1 = color_distance(home, away1)
    dist2 = color_distance(home, away2)
    if dist2 < dist1:
        away = away1
    else:
        away = away2
    return ('rgb' + home, 'rgb' + away)


def color_distance(color1, color2):
    c1 = color1.replace('(', '').replace(')', '').replace(' ', '').split(',')
    c2 = color2.replace('(', '').replace(')', '').replace(' ', '').split(',')
    dist = 0
    for i in range(3):
        dist = dist + abs(int(c1[i]) - int(c2[i]))
    return (dist)


color_distance('(0, 43, 92)', '(50, 24, 55)')

# In[73]:


app = Dash(__name__, external_stylesheets=external_stylesheets)
#app = JupyterDash(__name__,                 external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Store(id='game-store'),
    html.Label('Season'),
    dcc.Dropdown(
        id='season-dropdown', clearable=False,
        value='2022-23', options=[
            {'label': x, 'value': x} for x in
            [str(q) + '-' + str(q + 1)[-2:] for q in range(1982, datetime.today().year + 1)]
        ]),
    html.Div([
        html.Label('Game Date'),
        dcc.DatePickerSingle(id='dates-dropdown', date=datetime.today().strftime('%Y-%m-%d'))
    ]),
    html.Div([

        html.Label('Click on a Game for More Info!'),
        dash_table.DataTable(id='game-list',
                             columns=[
                                 {'id': 'home_name', 'name': 'Home'},
                                 {'id': 'away_name', 'name': 'Away'},
                                 {'id': 'home_score', 'name': 'Home Score'},
                                 {'id': 'away_score', 'name': 'Away Score'},
                                 {'id': 'prob', 'name': 'Probability of Home Win', 'type': 'numeric',
                                  'format': dash_table.FormatTemplate.percentage(1)}
                             ])
    ]),
    html.Hr(),
    html.Div([
        dash_table.DataTable(id='game-data',
                             columns=[
                                 {'id': 'description', 'name': 'Status'},
                                 {'id': 'home_score', 'name': 'Home Score'},
                                 {'id': 'away_score', 'name': 'Away Score'},
                                 {'id': 'period', 'name': 'Period'},
                                 {'id': 'game_clock', 'name': 'Game Clock'},
                                 {'id': 'prob', 'name': 'Probability of Home Win', 'type': 'numeric',
                                  'format': dash_table.FormatTemplate.percentage(1)}
                             ])
    ]),
    html.Hr(),

    html.Div([
        dcc.Graph(id='win-prob-graph', figure=blank_fig())
    ])

])


@app.callback(
    [Output('dates-dropdown', 'min_date_allowed'),
     Output('dates-dropdown', 'max_date_allowed'),
     Output('dates-dropdown', 'date'),
     Output('dates-dropdown', 'initial_visible_month'),
     Output('dates-dropdown', 'disabled_days')
     ],
    Input('season-dropdown', 'value'))
def set_days(s):
    days = get_days_by_season(s)

    min_date = min(days)
    max_date = max(days)
    disabled = []

    i = datetime.strptime(min_date, '%Y-%m-%d')

    while i <= datetime.strptime(max_date, '%Y-%m-%d'):

        i_format = datetime.strftime(i, '%Y-%m-%d')
        if i_format not in days:
            disabled.append(i_format)
        i = i + timedelta(days=1)

    return min_date, max_date, max_date, min_date, disabled


@app.callback(
    [Output('game-list', 'data'),
     Output('game-store', 'data')],
    Input('dates-dropdown', 'date')
)
def set_games(game_date):
    games = get_game_dict(datetime.strptime(game_date, '%Y-%m-%d'))
    data = [{**x, **{'id': i}} for i, x in games.items()]
    selection = list(games.keys())[0]
    return (data, data)


@app.callback(
    [
        Output('win-prob-graph', 'figure'),
        Output('game-data', 'data')
    ]
    ,
    [Input('game-list', 'active_cell'),
     Input('game-store', 'data')]
)
def update_game_table(active_cell, game_df):
    if not active_cell:
        return [blank_fig(), [{}]]

    game_id = active_cell['row_id']

    print(game_id)
    game_info = None
    for x in game_df:
        if x['id'] == game_id:
            game_info = x

    if game_info is None:
        return [blank_fig(), [{}]]

    home_color, away_color = get_colors(game_info['home_id'], game_info['away_id'])

    g = get_game_df(game_id, full=True)
    g.fillna('')
    g['description'] = g['description'].fillna('Current').astype('str')

    # Small table with quarterly info
    if (len(g.loc[g.description.str.slice(stop=6) == 'End of', :]) > 0):
        data = g.loc[g.description.str.slice(stop=6) == 'End of', :]
    else:
        data = g

    periods = max(g['period'])
    ticks = [0]
    labs = ['Start of Game']
    tot = 0
    for x in range(periods):
        if x < 4:
            tot = tot + (12 * 60)
        else:
            tot = tot + (5 * 60)
        ticks.append(tot)

        if periods == x + 1:
            labs.append('End of Game')
        elif x < 3:
            labs.append('End of Period ' + str(x + 1))
        elif x == 3:
            labs.append('End of Regulation')
        else:
            labs.append('End of Overtime ' + str(x - 3))

    g = g.rename(columns={'prob': 'Probability of Home Win',
                          'prob_category': 'General Effect of Play',
                          'prob_diff': 'Specific Win Probability Effect',
                          'description': 'Description of Play'})

    g['game_time'] = g.apply(lambda x: reverse_seconds(x['game_clock'], x['period']), axis=1)

    #     ['away_score', 'home_score', 'game_clock', 'period', 'Probability of Home Win', 'Specific Win Probability Effect', 'General Effect of Play', 'Description of Play']

    fig = px.scatter(g, y='Probability of Home Win'
                     , x='game_time'
                     , hover_name='Description of Play'
                     , hover_data={
            'game_time': False,
            'Probability of Home Win': True,
            'General Effect of Play': True,
            'Specific Win Probability Effect': True,
            'Description of Play': True
        }
                     )
    fig.update_traces(hovertemplate="""%{customdata[2]}
        <br>Probability of Home Win=%{y}
        <br>General Effect of Play=%{customdata[0]}
        <br>Specific Win Probability Effect=%{customdata[1]:~%}<extra></extra>"""
                      )
    fig.update_layout(
        xaxis_title=None,

        xaxis_tickvals=ticks,
        xaxis_ticktext=labs,
        yaxis_tickvals=[0, 0.25, 0.5, 0.75, 1],
        yaxis_range=[0, 1],
        yaxis_tickformat=',.0%'
    )
    fig.update_traces(mode='lines')

    g_home = g.loc[g['General Effect of Play'] == 'Home Very Positive', :]
    g_away = g.loc[g['General Effect of Play'] == 'Away Very Positive', :]

    fig.add_trace(
        go.Scatter(
            y=g_home['Probability of Home Win'],
            x=g_home['game_time'],
            name='home_big',
            customdata=np.stack((
                g_home['General Effect of Play'],
                g_home['Specific Win Probability Effect'],
                g_home['Description of Play']), axis=-1)

        )
    )

    fig.add_trace(
        go.Scatter(
            y=g_away['Probability of Home Win'],
            x=g_away['game_time'],
            name='away_big',
            customdata=np.stack((
                g_away['General Effect of Play'],
                g_away['Specific Win Probability Effect'],
                g_away['Description of Play']), axis=-1)

        )
    )

    fig.update_traces(marker=dict(size=15, color=home_color), mode='markers', selector=dict(name="home_big"),
                      name=game_info['home_name'],
                      hovertemplate="""%{customdata[2]}
        <br>Probability of Home Win=%{y}
        <br>General Effect of Play=%{customdata[0]}
        <br>Specific Win Probability Effect=%{customdata[1]:~%}<extra></extra>"""
                      )
    fig.update_traces(marker=dict(size=15, color=away_color), mode='markers', selector=dict(name="away_big"),
                      name=game_info['away_name'],
                      hovertemplate="""%{customdata[2]}
        <br>Probability of Home Win=%{y}
        <br>General Effect of Play=%{customdata[0]}
        <br>Specific Win Probability Effect=%{customdata[1]:~%}<extra></extra>"""
                      )

    return ([fig, data.to_dict('records')])


#app.run_server(mode='inline')

if __name__ == '__main__':
    app.run_server(debug=True)
