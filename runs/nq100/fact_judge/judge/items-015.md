Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 5 items.

=== ITEM id=a4288dcd8924d478 ===
Gold sentences and facts:
Sentence 0: Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington , D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game - winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of seven to win the series when trailing three games to one . 1926 St. Louis Cardinals New York Yankees 3 -- 2 Yankee Stadium , the Bronx , New York City Grover Cleveland Alexander strikes out Tony Lazzeri with the bases loaded to end the seventh and preserve the Cardinals lead . Two innings later Babe Ruth is caught stealing to end the World Series . 1931 St. Louis Cardinals Philadelphia Athletics 4 -- 2 St. Louis , Missouri Last postseason game for the Athletics while based in Philadelphia . 1934 St. Louis Cardinals Detroit Tigers 11 -- 0 Navin Field , Detroit Dizzy Dean 's second win , a six - hit shutout on one day of rest , gives the Cardinals the title . 1940 Cincinnati Reds Detroit Tigers 2 -- 1 Crosley Field , Cincinnati This game was over in a snappy 1 : 47 while a sacrifice bunt and sacrifice fly scored the winning runs . 1945 Detroit Tigers Chicago Cubs 9 -- 3 Wrigley Field , Chicago Cubs last World Series appearance until 2016 . See also : Curse of the Billy Goat 1946 St. Louis Cardinals Boston Red Sox 4 -- 3 Sportsman 's Park III , St. Louis , Missouri Enos Slaughter 's famous mad dash scored the deciding run to win the game and the series . 1947 New York Yankees Brooklyn Dodgers 5 -- 2 Yankee Stadium , the Bronx , New York City This was the Series that featured Cookie Lavagetto 's pinch - hit that spoiled Bill Bevens ' no - hit bid and Al Gionfriddo 's catch to rob Joe DiMaggio of a home run . 1952 New York Yankees Brooklyn Dodgers 4 -- 2 Ebbets Field , Brooklyn , New York City Billy Martin 's game saving infield catch preserves the Yankees lead and helps win the game and the series . 1955 Brooklyn Dodgers New York Yankees 2 -- 0 Yankee Stadium , the Bronx , New York City Dodgers ' only championship in Brooklyn . The home team won all games of the series , except for game seven . 1956 New York Yankees Brooklyn Dodgers 9 -- 0 Ebbets Field , Brooklyn , New York City Jackie Robinson 's final Major League game . Last postseason game at Ebbets Field and Dodgers ' last postseason game before moving to Los Angeles . The home team won all games of the series , except for game seven . 1957 Milwaukee Braves New York Yankees 5 -- 0 Yankee Stadium , the Bronx , New York City The Braves ' only championship in Milwaukee . 1958 New York Yankees Milwaukee Braves 6 -- 2 County Stadium , Milwaukee The Yankees win their 18th World Series title after being down three games to one . This was the Braves ' last postseason game in Milwaukee . 1960 Pittsburgh Pirates New York Yankees 10 -- 9 Forbes Field , Pittsburgh Bill Mazeroski hit the walk - off home run to win the Pirates the championship . Last postseason game in Forbes Field . Last postseason game for Yankees ' manager Casey Stengel . Highest scoring game seven . 1962 New York Yankees San Francisco Giants 1 -- 0 Candlestick Park , San Francisco With runners on second and third and two outs , Willie McCovey 's line drive to Bobby Richardson ended the game . First game seven that ended with a 1 - 0 score . 1964 St. Louis Cardinals New York Yankees 7 -- 5 Busch Stadium I , St. Louis , Missouri The Yankees ' last World Series appearance until 1976 . Last postseason game at Busch Stadium I . 1965 Los Angeles Dodgers Minnesota Twins 2 -- 0 Metropolitan Stadium , Bloomington , Minnesota Pitching through immense pain and on two days rest , Sandy Koufax throws a 3 hit shutout to win the Series and World Series MVP . The home team won all games of the series , except for game seven . 1967 St. Louis Cardinals Boston Red Sox 7 -- 2 Fenway Park , Boston Bob Gibson wins his third game of the Series and also homers . 1968 Detroit Tigers St. Louis Cardinals 4 -- 1 Busch Stadium II , St. Louis , Missouri Mickey Lolich out - duels Bob Gibson . The Tigers rallied from a 3 - 1 deficit to win the series . First game seven loss by the Cardinals in the World Series . 1971 Pittsburgh Pirates Baltimore Orioles 2 -- 1 Memorial Stadium , Baltimore The home team won all games of the series , except for game seven . 1972 Oakland Athletics Cincinnati Reds 3 -- 2 Riverfront Stadium , Cincinnati The A 's first championship in Oakland . The A 's were the last American League team to win a World Series game seven on the road until 2017 . 1973 Oakland Athletics New York Mets 5 -- 2 Oakland -- Alameda County Coliseum , Oakland , California Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game to set the tone for victory and the series . Cincinnati Reds Boston Red Sox 4 -- 3 Fenway Park , Boston One night after Carlton Fisk 's famous home run , the Reds rebound to take the Series . 1979 Pittsburgh Pirates Baltimore Orioles 4 -- 1 Memorial Stadium , Baltimore The Pittsburgh Pirates , powered by NLCS and World Series MVP Willie Stargell , rally from a 3 - 1 deficit to win the series . The Pirates were the last team to win a road game seven until 2014 . 1982 St. Louis Cardinals Milwaukee Brewers 6 -- 3 Busch Stadium II , St. Louis , Missouri Joaquín Andújar earns his second win of the Series and Bruce Sutter his second save . 1985 Kansas City Royals St. Louis Cardinals 11 -- 0 Royals Stadium , Kansas City , Missouri Second World Series held entirely in the state of Missouri . The Royals also won the 1985 American League Championship Series in seven games . Royals became first team to win the World Series after losing the first two games at home . The Royals were the last team to come back from 3 -- 1 down to win a title , in any of the five major men 's professional sports leagues , until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3 -- 1 . The 1985 Royals were the last team to win a World Series after trailing 3 -- 1 until the Cubs did so in 2016 . 1986 New York Mets Boston Red Sox 8 -- 5 Shea Stadium , Flushing , Queens , New York Mets become the first team to be one strike away from elimination and come back to win the series . They also become the second team to win the series after losing the first two at home . Minnesota Twins St. Louis Cardinals 4 -- 2 Hubert H. Humphrey Metrodome , Minneapolis The home team won all seven games of the series for the first time in MLB history . 1991 Minnesota Twins Atlanta Braves 1 -- 0 F / 10 Hubert H. Humphrey Metrodome , Minneapolis Like in 1987 , the home team won all games of the series . The Twins won in the bottom of the 10th on a walk - off single by Gene Larkin scoring Dan Gladden after ​ 9 ⁄ scoreless innings of play . Jack Morris goes all ten innings for the Twins and is named the Series MVP . Twins Centerfielder Kirby Puckett hit a famous walk - off home run in the bottom of the eleventh inning of the previous night 's game to force this decisive game . 1997 Florida Marlins Cleveland Indians 3 -- 2 F / 11 Pro Player Stadium , Miami Gardens , Florida Florida becomes the first wild card team to win the World Series . The Marlins rallied for one run in the bottom of the ninth to tie and won in the eleventh on an Édgar Rentería RBI single off Charles Nagy . 2001 Arizona Diamondbacks New York Yankees 3 -- 2 Bank One Ballpark , Phoenix , Arizona Arizona becomes the fastest expansion team to win the World Series , rallying for two runs in the bottom of the ninth for a come - from - behind win , capped off by Luis Gonzalez hitting the winning RBI single off closer Mariano Rivera . The home team won every game of the series . 2002 Anaheim Angels San Francisco Giants 4 -- 1 Edison International Field , Anaheim , California Both World Series teams were wild card entrants into the playoffs . John Lackey became the second rookie pitcher to win a World Series game seven . 2011 St. Louis Cardinals Texas Rangers 6 -- 2 Busch Stadium , St. Louis , Missouri Cardinals became the second team to be one strike away from elimination and come back to win the series , which they did twice . Just like the 1991 series , a walk - off home run in the bottom of the eleventh inning of game six , forced this decisive game with David Freese providing the fire power . Most recent game seven won by the home team . 2014 San Francisco Giants Kansas City Royals 3 -- 2 Kauffman Stadium , Kansas City , Missouri Like in 2002 , both World Series teams were wild card entrants into the playoffs . The Giants became the first NL team to win a game seven on the road since 1979 . NLCS MVP and eventual World Series MVP Madison Bumgarner , normally a starter , pitched five scoreless innings of relief on two days rest to save the Giants ' third title in five years . 2016 Chicago Cubs Cleveland Indians 8 -- 7 F / 10 Progressive Field , Cleveland , Ohio Chicago Cubs ' first World Series win since 1908 . Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City Royals in 1985 .
  Fact 0.0: In the 1909 World Series the Pittsburgh Pirates beat the Detroit Tigers 8-0 at Bennett Park in Detroit.
  Fact 0.1: Rookie Babe Adams threw a six-hit shutout for his third win of the series in 1909.
  Fact 0.2: The Boston Red Sox beat the New York Giants 3-2 in 10 innings at Fenway Park in Boston in 1912.
  Fact 0.3: The 1912 decisive contest was game eight, because game two had ended in a tie.
  Fact 0.4: The Washington Senators beat the New York Giants 4-3 in 12 innings at Griffith Stadium in Washington, D.C. in 1924.
  Fact 0.5: The 1924 game was the longest game seven in innings.
  Fact 0.6: In the 12th inning of the 1924 game, Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy.
  Fact 0.7: Ruel scored on a game-winning RBI double by Earl McNeely.
  Fact 0.8: The Pittsburgh Pirates beat the Washington Senators 9-7 at Forbes Field in Pittsburgh in 1925.
  Fact 0.9: Pittsburgh became the first team in a best-of-seven series to win the series after trailing three games to one.
  Fact 0.10: The St. Louis Cardinals beat the New York Yankees 3-2 at Yankee Stadium in the Bronx, New York City in 1926.
  Fact 0.11: Grover Cleveland Alexander struck out Tony Lazzeri with the bases loaded to end the seventh inning and preserve the Cardinals lead in 1926.
  Fact 0.12: Two innings later in 1926, Babe Ruth was caught stealing to end the World Series.
  Fact 0.13: The St. Louis Cardinals beat the Philadelphia Athletics 4-2 in St. Louis, Missouri in 1931.
  Fact 0.14: The 1931 game was the last postseason game for the Athletics while they were based in Philadelphia.
  Fact 0.15: The St. Louis Cardinals beat the Detroit Tigers 11-0 at Navin Field in Detroit in 1934.
  Fact 0.16: Dizzy Dean earned his second win of the 1934 Series with a six-hit shutout on one day of rest.
  Fact 0.17: The 1934 win gave the Cardinals the title.
  Fact 0.18: The Cincinnati Reds beat the Detroit Tigers 2-1 at Crosley Field in Cincinnati in 1940.
  Fact 0.19: The 1940 game was over in a snappy 1 hour 47 minutes.
  Fact 0.20: A sacrifice bunt and a sacrifice fly scored the winning runs in the 1940 game.
  Fact 0.21: The Detroit Tigers beat the Chicago Cubs 9-3 at Wrigley Field in Chicago in 1945.
  Fact 0.22: The Chicago Cubs' 1945 appearance was their last World Series appearance until 2016.
  Fact 0.23: The 1945 entry includes a see-also reference to the Curse of the Billy Goat.
  Fact 0.24: The St. Louis Cardinals beat the Boston Red Sox 4-3 at Sportsman's Park III in St. Louis, Missouri in 1946.
  Fact 0.25: Enos Slaughter's famous mad dash scored the deciding run to win the 1946 game and the series.
  Fact 0.26: The New York Yankees beat the Brooklyn Dodgers 5-2 at Yankee Stadium in the Bronx, New York City in 1947.
  Fact 0.27: The 1947 Series featured Cookie Lavagetto's pinch-hit that spoiled Bill Bevens' no-hit bid.
  Fact 0.28: The 1947 Series featured Al Gionfriddo's catch that robbed Joe DiMaggio of a home run.
  Fact 0.29: The New York Yankees beat the Brooklyn Dodgers 4-2 at Ebbets Field in Brooklyn, New York City in 1952.
  Fact 0.30: Billy Martin's game-saving infield catch in 1952 preserved the Yankees lead and helped win the game and the series.
  Fact 0.31: The Brooklyn Dodgers beat the New York Yankees 2-0 at Yankee Stadium in the Bronx, New York City in 1955.
  Fact 0.32: The 1955 championship was the Dodgers' only championship in Brooklyn.
  Fact 0.33: In the 1955 series the home team won all games except game seven.
  Fact 0.34: The New York Yankees beat the Brooklyn Dodgers 9-0 at Ebbets Field in Brooklyn, New York City in 1956.
  Fact 0.35: The 1956 game was Jackie Robinson's final Major League game.
  Fact 0.36: The 1956 game was the last postseason game at Ebbets Field.
  Fact 0.37: The 1956 game was the Dodgers' last postseason game before moving to Los Angeles.
  Fact 0.38: In the 1956 series the home team won all games except game seven.
  Fact 0.39: The Milwaukee Braves beat the New York Yankees 5-0 at Yankee Stadium in the Bronx, New York City in 1957.
  Fact 0.40: The 1957 championship was the Braves' only championship in Milwaukee.
  Fact 0.41: The New York Yankees beat the Milwaukee Braves 6-2 at County Stadium in Milwaukee in 1958.
  Fact 0.42: The 1958 Yankees won their 18th World Series title after being down three games to one.
  Fact 0.43: The 1958 game was the Braves' last postseason game in Milwaukee.
  Fact 0.44: The Pittsburgh Pirates beat the New York Yankees 10-9 at Forbes Field in Pittsburgh in 1960.
  Fact 0.45: Bill Mazeroski hit the walk-off home run in 1960 to win the championship for the Pirates.
  Fact 0.46: The 1960 game was the last postseason game in Forbes Field.
  Fact 0.47: The 1960 game was the last postseason game for Yankees manager Casey Stengel.
  Fact 0.48: The 1960 game was the highest scoring game seven.
  Fact 0.49: The New York Yankees beat the San Francisco Giants 1-0 at Candlestick Park in San Francisco in 1962.
  Fact 0.50: With runners on second and third and two outs in 1962, Willie McCovey's line drive to Bobby Richardson ended the game.
  Fact 0.51: The 1962 game was the first game seven to end with a 1-0 score.
  Fact 0.52: The St. Louis Cardinals beat the New York Yankees 7-5 at Busch Stadium I in St. Louis, Missouri in 1964.
  Fact 0.53: The 1964 Yankees' appearance was their last World Series appearance until 1976.
  Fact 0.54: The 1964 game was the last postseason game at Busch Stadium I.
  Fact 0.55: The Los Angeles Dodgers beat the Minnesota Twins 2-0 at Metropolitan Stadium in Bloomington, Minnesota in 1965.
  Fact 0.56: Sandy Koufax pitched through immense pain on two days' rest in 1965 and threw a 3-hit shutout.
  Fact 0.57: Sandy Koufax won the 1965 Series and was named World Series MVP.
  Fact 0.58: In the 1965 series the home team won all games except game seven.
  Fact 0.59: The St. Louis Cardinals beat the Boston Red Sox 7-2 at Fenway Park in Boston in 1967.
  Fact 0.60: Bob Gibson won his third game of the 1967 Series and also hit a home run.
  Fact 0.61: The Detroit Tigers beat the St. Louis Cardinals 4-1 at Busch Stadium II in St. Louis, Missouri in 1968.
  Fact 0.62: Mickey Lolich outdueled Bob Gibson in the 1968 game.
  Fact 0.63: The Detroit Tigers rallied from a 3-1 deficit to win the 1968 series.
  Fact 0.64: The 1968 game was the first game seven loss by the Cardinals in the World Series.
  Fact 0.65: The Pittsburgh Pirates beat the Baltimore Orioles 2-1 at Memorial Stadium in Baltimore in 1971.
  Fact 0.66: In the 1971 series the home team won all games except game seven.
  Fact 0.67: The Oakland Athletics beat the Cincinnati Reds 3-2 at Riverfront Stadium in Cincinnati in 1972.
  Fact 0.68: The 1972 championship was the first championship for the Athletics in Oakland.
  Fact 0.69: The Athletics were the last American League team to win a World Series game seven on the road until 2017.
  Fact 0.70: The Oakland Athletics beat the New York Mets 5-2 at the Oakland-Alameda County Coliseum in Oakland, California in 1973.
  Fact 0.71: Bert Campaneris and Reggie Jackson hit the only Oakland home runs in the 1973 game, setting the tone for victory and the series.
  Fact 0.72: The Cincinnati Reds beat the Boston Red Sox 4-3 at Fenway Park in Boston in 1973.
  Fact 0.73: The 1973 Reds game came one night after Carlton Fisk's famous home run.
  Fact 0.74: The Cincinnati Reds rebounded to take the 1973 Series.
  Fact 0.75: The Pittsburgh Pirates beat the Baltimore Orioles 4-1 at Memorial Stadium in Baltimore in 1979.
  Fact 0.76: The Pittsburgh Pirates, powered by NLCS and World Series MVP Willie Stargell, rallied from a 3-1 deficit to win the 1979 series.
  Fact 0.77: The Pittsburgh Pirates were the last team to win a road game seven until 2014.
  Fact 0.78: The St. Louis Cardinals beat the Milwaukee Brewers 6-3 at Busch Stadium II in St. Louis, Missouri in 1982.
  Fact 0.79: Joaquin Andujar earned his second win of the 1982 Series.
  Fact 0.80: Bruce Sutter earned his second save of the 1982 Series.
  Fact 0.81: The Kansas City Royals beat the St. Louis Cardinals 11-0 at Royals Stadium in Kansas City, Missouri in 1985.
  Fact 0.82: The 1985 Series was the second World Series held entirely in the state of Missouri.
  Fact 0.83: The Royals also won the 1985 American League Championship Series in seven games.
  Fact 0.84: The Royals became the first team to win the World Series after losing the first two games at home.
  Fact 0.85: The Royals were the last team to come back from 3-1 down to win a title in any of the five major men's professional sports leagues until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3-1.
  Fact 0.86: The 1985 Royals were the last team to win a World Series after trailing 3-1 until the Cubs did so in 2016.
  Fact 0.87: The New York Mets beat the Boston Red Sox 8-5 at Shea Stadium in Flushing, Queens, New York in 1986.
  Fact 0.88: The 1986 Mets became the first team to be one strike away from elimination and come back to win the series.
  Fact 0.89: The 1986 Mets became the second team to win the series after losing the first two games at home.
  Fact 0.90: The Minnesota Twins beat the St. Louis Cardinals 4-2 at the Hubert H. Humphrey Metrodome in Minneapolis in the entry listed without a year.
  Fact 0.91: The home team won all seven games of that series for the first time in MLB history.
  Fact 0.92: The Minnesota Twins beat the Atlanta Braves 1-0 in 10 innings at the Hubert H. Humphrey Metrodome in Minneapolis in 1991.
  Fact 0.93: Like in 1987, the home team won all games of the 1991 series.
  Fact 0.94: The Twins won the 1991 game in the bottom of the 10th on a walk-off single by Gene Larkin, scoring Dan Gladden after scoreless innings of play.
  Fact 0.95: Jack Morris went all ten innings for the Twins in 1991 and was named Series MVP.
  Fact 0.96: Kirby Puckett hit a famous walk-off home run in the bottom of the 11th inning of the previous night's game to force the decisive 1991 game.
  Fact 0.97: The Florida Marlins beat the Cleveland Indians 3-2 in 11 innings at Pro Player Stadium in Miami Gardens, Florida in 1997.
  Fact 0.98: Florida became the first wild card team to win the World Series in 1997.
  Fact 0.99: The Marlins rallied for one run in the bottom of the ninth in 1997 to tie the game.
  Fact 0.100: The Marlins won the 1997 game in the 11th inning on an Edgar Renteria RBI single off Charles Nagy.
  Fact 0.101: The Arizona Diamondbacks beat the New York Yankees 3-2 at Bank One Ballpark in Phoenix, Arizona in 2001.
  Fact 0.102: Arizona became the fastest expansion team to win the World Series in 2001.
  Fact 0.103: Arizona rallied for two runs in the bottom of the ninth in 2001 for a come-from-behind win.
  Fact 0.104: Luis Gonzalez hit the winning RBI single off closer Mariano Rivera in 2001.
  Fact 0.105: The home team won every game of the 2001 series.
  Fact 0.106: The Anaheim Angels beat the San Francisco Giants 4-1 at Edison International Field in Anaheim, California in 2002.
  Fact 0.107: Both 2002 World Series teams were wild card entrants into the playoffs.
  Fact 0.108: John Lackey became the second rookie pitcher to win a World Series game seven in 2002.
  Fact 0.109: The St. Louis Cardinals beat the Texas Rangers 6-2 at Busch Stadium in St. Louis, Missouri in 2011.
  Fact 0.110: The Cardinals became the second team to be one strike away from elimination and come back to win the series, which they did twice.
  Fact 0.111: Like the 1991 series, a walk-off home run in the bottom of the 11th inning of game six forced the 2011 decisive game, with David Freese providing the firepower.
  Fact 0.112: The 2011 game seven was the most recent game seven won by the home team.
  Fact 0.113: The San Francisco Giants beat the Kansas City Royals 3-2 at Kauffman Stadium in Kansas City, Missouri in 2014.
  Fact 0.114: Like in 2002, both 2014 World Series teams were wild card entrants into the playoffs.
  Fact 0.115: The Giants became the first NL team to win a game seven on the road since 1979.
  Fact 0.116: Madison Bumgarner was the NLCS MVP and eventual World Series MVP in 2014.
  Fact 0.117: Madison Bumgarner is normally a starter but pitched five scoreless innings of relief on two days' rest in 2014.
  Fact 0.118: The 2014 relief pitching saved the Giants' third title in five years.
  Fact 0.119: The Chicago Cubs beat the Cleveland Indians 8-7 in 10 innings at Progressive Field in Cleveland, Ohio in 2016.
  Fact 0.120: The 2016 win was the Chicago Cubs' first World Series win since 1908.
  Fact 0.121: The 2016 game was played between the two teams with the longest active championship droughts.
  Fact 0.122: The Cubs are the first Major League Baseball team to come back to win the World Series from a 3-1 series deficit since the 1985 Royals.
  Fact 0.123: The 2016 game seven was the most recent World Series game seven to be decided in extra innings.
  Fact 0.124: The 2016 game seven was the first extra innings game seven to be won by the road team.
  Fact 0.125: The Houston Astros beat the Los Angeles Dodgers 5-1 at Dodger Stadium in Los Angeles in 2017.
  Fact 0.126: The 2017 series was the first instance of back-to-back game sevens since 2001-2002.
  Fact 0.127: The 2017 championship was the first World Series championship for the Astros.
  Fact 0.128: The Astros became the first AL team to win a World Series game 7 on the road since 1972.
  Fact 0.129: The Astros became the second team to win two winner-take-all game sevens in the same postseason since the Kansas City Royals in 1985.

Retrieved items:
<item 1>
Game seven
Louis the NL Pennant . The Cardinals would lose the World Series as the Minnesota Twins give new meaning to the term home - field advantage in the World Series
as only the decisive game seven is close . 1988 NLCS Los Angeles Dodgers New York Mets 6 -- 0 Dodger Stadium , Los Angeles Orel Hershiser pitches a five - hit
shutout to send the Dodgers to the World Series . Until 2017 , this was the only game seven of any postseason series to be played in Dodger Stadium . 1991 NLCS
Atlanta Braves Pittsburgh Pirates 4 -- 0 Three Rivers Stadium , Pittsburgh , Pennsylvania Back - to - back shutouts in games six and seven helps the Braves win
the pennant . The Braves would lose the series to Minnesota in seven games in what is regarded as the best World Series of all time . 1992 NLCS Atlanta Braves
Pittsburgh Pirates 3 -- 2 Atlanta -- Fulton County Stadium , Atlanta Atlanta scored all three runs in the bottom of the ninth . Pinch hitter Francisco Cabrera
drove in the two winning runs with a two - out single with the bases loaded . The winning run was scored by Sid Bream .
</item 1>

<item 2>
Game seven
2006 NLCS St. Louis Cardinals New York Mets 3 -- 1 Shea Stadium , Flushing , Queens , New York Last postseason game at Shea Stadium . Yadier Molina hits a two -
run homer in the top of the ninth and Adam Wainwright strikes out Carlos Beltrán in the bottom of the ninth with the bases loaded to send the Cardinals to the
World Series where they would beat Detroit in five games . Most recent game seven in either LCS to be won by the road team . 2007 ALCS Boston Red Sox Cleveland
Indians 11 -- 2 Fenway Park , Boston , Massachusetts Boston overcomes a 3 - games - to - 1 deficit to win the pennant and would win the second World Series
championship in four years . Like four years ago , the result is a sweep , this time over the Colorado Rockies . 2008 ALCS Tampa Bay Rays Boston Red Sox 3 -- 1
Tropicana Field , St. Petersburg , Florida Matt Garza pitches Tampa Bay to its first American League Championship . The Rays would end up losing to the
Philadelphia Phillies in five games . 2012 NLCS San Francisco Giants St.
</item 2>

<item 3>
Game seven
Home teams are 19 -- 20 in game sevens . Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit
Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This
decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington
, D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game -
winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of
seven to win the series when trailing three games to one . 1926 St.
</item 3>

<item 4>
Game seven
League Championship Series that were decided on a game seven : Year Series Winning team Losing team Score Site Notes 1985 ALCS Kansas City Royals Toronto Blue
Jays 6 -- 2 Exhibition Stadium , Toronto The Royals rallied from a 3 - 1 deficit , including taking the last two on the road to win the pennant . They would go
on to win the 1985 World Series in seven games , becoming the first team in MLB history to win two game sevens in a single postseason . 1986 ALCS Boston Red Sox
California Angels 8 -- 1 Fenway Park , Boston , Massachusetts Dave Henderson 's famous home run off of Donnie Moore turned the tide as Boston rallied from a 3 -
1 deficit to win the series . Boston would lose the 1986 World Series in seven games . NLCS St. Louis Cardinals San Francisco Giants 6 -- 0 Busch Stadium ( II )
, St. Louis , Missouri Although Jeffrey Leonard would homer in the first four games of the series it was not enough as back - to - back shutouts by the
Cardinals in games six and seven gave St. Louis the NL Pennant .
</item 4>

<item 5>
Game seven
Major League Baseball Advanced Media . November 2 , 2017 . Retrieved November 6 , 2017 . Jump up ^ `` 1909 World Series '' . MLB.com . Retrieved January 30 ,
2009 . Jump up ^ `` 1912 World Series '' . MLB.com . Retrieved January 30 , 2009 . Jump up ^ `` 1924 World Series '' . MLB.com . Retrieved January 30 , 2009 .
Jump up ^ `` 1925 World Series '' . MLB.com . Retrieved January 30 , 2009 . Jump up ^ `` 1926 World Series '' . MLB.com . Retrieved January 30 , 2009 . Jump up
^ `` 1931 World Series '' . MLB.com . Retrieved January 30 , 2009 . Jump up ^ `` 1934 World Series '' . MLB.com . Retrieved January 30 , 2009 . Jump up ^ ``
1940 World Series '' . MLB.com . Retrieved January 30 , 2009 .
</item 5>

<item 6>
Game seven
2012 NLCS San Francisco Giants St. Louis Cardinals 9 -- 0 AT&T Park , San Francisco , California First Giants victory in a winner - take - all game seven . The
Giants would go on to sweep the Detroit Tigers in the World Series . 2017 ALCS Houston Astros New York Yankees 4 -- 0 Minute Maid Park , Houston , Texas The
home team won all games of the series . Houston became the first franchise to win the pennant in both the National League and American League . The Astros would
go on to win the World Series over the Los Angeles Dodgers in seven games , becoming the second MLB team to win two game sevens in a single postseason .
National Hockey League ( edit ) Main article : List of NHL game sevens In the NHL playoffs , game seven can occur in all playoff series . In the Stanley Cup
Final , the NHL gained complete control of the Cup in 1926 , and the league instituted the best - of - seven series starting in the 1938 -- 39 season . No
Stanley Cup Final game seven has ever ended with a 1 -- 0 score .
</item 6>

<item 7>
Game seven
1 World Series 1.1. 2 League Championship Series 1.2 National Hockey League 1.3 National Basketball Association 1.4 Chinese Professional Baseball League 2
Comebacks 3 References Examples ( edit ) Major League Baseball ( edit ) In the MLB Postseason , game seven can occur in the League Championship Series and the
World Series ( the divisional series are best of five ) . In the World Series , there have been 39 decisive game sevens ( of the four World Series that were
contested on a best - of - nine basis , none went to a game nine , though all four included a non-decisive game seven ) . World Series ( edit ) World Series
that were decided on a game seven : All extra inning game sevens involve three franchises each with two appearances , the New York / San Francisco Giants (
losing both ) , the Washington Senators / Minnesota Twins ( winning both ) and the Cleveland Indians ( losing both ) . Home teams are 19 -- 20 in game sevens .
</item 7>

<item 8>
Game seven
1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999
2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most
Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV
ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World
Series Stanley Cup Finals Stanley Cup awarded on challenge basis 1893 --
</item 8>

<item 9>
Game seven
Most codes of football do not employ a best - of - seven series ( or any best - of - x series in general ) , hence game sevens are not played in those leagues .
Some playoff rounds ( such as MLB 's current Division Series ) are played in a best of five format , such that game five has similar qualities to those
described above , though the suspense and drama have less time to build in a shorter series . Furthermore , the World Series of 1903 , 1919 , 1920 , and 1921
were played in a best of nine format , though none of the four went to a decisive game nine . The game seven is comparable to a final or to a single game in a
single - elimination tournament or to a one - game playoff . The game is equivalent to the Super Bowl game in the National Football League in that the game 's
winner is the league 's champion for the season . Contents ( hide ) 1 Examples 1.1 Major League Baseball 1.1. 1 World Series 1.1.
</item 9>

<item 10>
Game seven
Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League
Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in
extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First
instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World
Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City
Royals in 1985 . League Championship Series ( edit ) The League Championship Series were originally decided best of five from 1969 until 1984 . Since 1985 when
both were expanded to best of seven contests , 16 series ( 7 American , 9 National ) were decided in game seven .
</item 10>
=== END ITEM id=a4288dcd8924d478 ===

=== ITEM id=7954a042a9de05c4 ===
Gold sentences and facts:
Sentence 0: Adobe Flash Player Original author ( s ) FutureWave Macromedia Developer ( s ) Adobe Systems Initial release 1996 ; 22 years ago ( 1996 ) Stable release ( s ) ( ± ) Windows , macOS , Linux , Chrome OS 28.0. 0.137 / January 9 , 2018 ; 16 days ago ( 2018 - 01 - 09 ) Android 4.0. x 11.1. 115.81 / September 10 , 2013 ; 4 years ago ( 2013 - 09 - 10 ) Android 2. x and 3. x 11.1. 111.73 / September 10 , 2013 ; 4 years ago ( 2013 - 09 - 10 ) Solaris 11.2. 202.223 / March 28 , 2012 ; 5 years ago ( 2012 - 03 - 28 ) Preview release ( s ) ( ± ) Windows , macOS , Linux , Chrome OS 28.0. 0.152 Beta / January 24 , 2018 ; 1 day ago ( 2018 - 01 - 24 ) Written in C++ Operating system Windows , macOS , Linux , Chrome OS , Solaris , BlackBerry Tablet OS , Android , and Pocket PC Platform Web browsers and ActiveX - based software Available in Chinese Simplified , Chinese Traditional , English , French , German , Italian , Japanese , Polish , Russian , Spanish , Korean , and Turkish Type Run - time environment , Media player , and Browser extension License Freeware Website www.adobe.com/products/flashplayer.html
  Fact 0.0: Adobe Flash Player's original authors were FutureWave and Macromedia.
  Fact 0.1: Adobe Systems is the developer of Adobe Flash Player.
  Fact 0.2: Adobe Flash Player had its initial release in 1996.
  Fact 0.3: The stable release 28.0.0.137 for Windows, macOS, Linux and Chrome OS is dated January 9, 2018.
  Fact 0.4: The Android 4.0.x stable release is version 11.1.115.81, dated September 10, 2013.
  Fact 0.5: The Android 2.x and 3.x stable release is version 11.1.111.73, dated September 10, 2013.
  Fact 0.6: The Solaris stable release is version 11.2.202.223, dated March 28, 2012.
  Fact 0.7: The preview release 28.0.0.152 Beta for Windows, macOS, Linux and Chrome OS is dated January 24, 2018.
  Fact 0.8: Adobe Flash Player is written in C++.
  Fact 0.9: Adobe Flash Player operates on Windows, macOS, Linux, Chrome OS, Solaris, BlackBerry Tablet OS, Android, and Pocket PC.
  Fact 0.10: Adobe Flash Player's platform is web browsers and ActiveX-based software.
  Fact 0.11: Adobe Flash Player is available in Chinese Simplified, Chinese Traditional, English, French, German, Italian, Japanese, Polish, Russian, Spanish, Korean, and Turkish.
  Fact 0.12: Adobe Flash Player's type is run-time environment, media player, and browser extension.
  Fact 0.13: Adobe Flash Player's license is freeware.
  Fact 0.14: Adobe Flash Player's website is www.adobe.com/products/flashplayer.html.
Sentence 1: Availability on desktop operating systems Platform Latest version Browser support Windows XP and later Windows Server 2003 and later 28.0. 0.137 Firefox , Chrome , Chromium , Safari , Opera , Internet Explorer , Microsoft Edge Windows 2000 11.1. 102.55 ? Windows 98 and ME 9.0. 115.0 ? Windows 95 and NT 4 7.0. 14.0 ? Mac OS X 10.6 or later 28.0. 0.137 Firefox , Chrome , Chromium , Safari , Opera Mac OS X 10.5 10.3. 183.90 ? Classic Mac OS , PowerPC 7.0. 14.0 ? Classic Mac OS , 68k 5.0 ? Linux 28.0. 0.137 Firefox , Chrome , Chromium , Opera Solaris 11.2. 202.223 ?
  Fact 1.0: On Windows XP and later and Windows Server 2003 and later, the latest Flash Player version is 28.0.0.137 with support for Firefox, Chrome, Chromium, Safari, Opera, Internet Explorer, and Microsoft Edge.
  Fact 1.1: On Windows 2000, the latest Flash Player version is 11.1.102.55, marked with a question mark as uncertain.
  Fact 1.2: On Windows 98 and ME, the latest Flash Player version is 9.0.115.0, marked with a question mark as uncertain.
  Fact 1.3: On Windows 95 and NT 4, the latest Flash Player version is 7.0.14.0, marked with a question mark as uncertain.
  Fact 1.4: On Mac OS X 10.6 or later, the latest Flash Player version is 28.0.0.137 with support for Firefox, Chrome, Chromium, Safari, and Opera.
  Fact 1.5: On Mac OS X 10.5, the latest Flash Player version is 10.3.183.90, marked with a question mark as uncertain.
  Fact 1.6: On Classic Mac OS with PowerPC, the latest Flash Player version is 7.0.14.0, marked with a question mark as uncertain.
  Fact 1.7: On Classic Mac OS with 68k, the latest Flash Player version is 5.0, marked with a question mark as uncertain.
  Fact 1.8: On Linux, the latest Flash Player version is 28.0.0.137 with support for Firefox, Chrome, Chromium, and Opera.
  Fact 1.9: On Solaris, the latest Flash Player version is 11.2.202.223, marked with a question mark as uncertain.

Retrieved items:
<item 1>
Adobe Flash Player
202.228 ) ( March 28 , 2012 ) Adobe Flash Player 11.2 ( version 11.2. 202.233 ) ( April 12 , 2012 ) Adobe Flash Player 11.2 ( version 11.2. 202.235 , codenamed
Brannan ) ( May 3 , 2012 ) The Windows version offers automatic updater options Dropped support of the browser plug - in for mobile devices ( Android ) .
Android app developers are encouraged to use Adobe Air and Android web developers should switch to HTML5 . Extended support for Flash player 11.2 on Solaris as
it is the last version to be supported . Adobe replaced Extended Support Release 11.2 on Linux with 24.0 on December 13 , 2016 . Adobe Flash Player 11.3 (
version 11.3. 300.257 ) ( June 8 , 2012 ) Adobe Flash Player 11.3 ( version 11.3. 300.262 ) ( June 21 , 2012 ) Adobe Flash Player 11.3 ( version 11.3.
</item 1>

<item 2>
Adobe Flash Player
0.152 , codenamed ( Market Street ( San Francisco ) Market ) ) ( September 9 , 2014 ) Improved support for browser zoom levels Adobe Flash Player 15 ( version
15.0. 0.167 ) ( September 24 , 2014 ) Adobe Flash Player 15 ( version 15.0. 0.223 ) ( November 11 , 2014 ) Adobe Flash Player 15 ( version 15.0. 0.239 ) (
November 25 , 2014 ) Adobe Flash Player 16 ( version 16.0. 0.235 , codenamed Natoma ) ( December 12 , 2014 ) Stage3D -- Standard Constrained Profile PPAPI
Installers for Windows and Mac Adobe Flash Player 16 ( version 16.0. 0.257 ) ( January 13 , 2015 ) Adobe Flash Player 16 ( version 16.0. 0.287 ) ( January 22 ,
2015 ) Adobe Flash Player 17 ( version 17.0.
</item 2>

<item 3>
Graph query row:
work: Adobe Flash Player
platform: macOS
version: None
fact: None
</item 3>

<item 4>
Graph query row:
work: Adobe Flash Player
platform: Windows XP
version: None
fact: None
</item 4>

<item 5>
Adobe Flash Player
Adobe Flash Player - wikipedia Adobe Flash Player Jump to : navigation , search Adobe Flash Player Original author ( s ) FutureWave Macromedia Developer ( s )
Adobe Systems Initial release 1996 ; 22 years ago ( 1996 ) Stable release ( s ) ( ± ) Windows , macOS , Linux , Chrome OS 28.0. 0.137 / January 9 , 2018 ; 16
days ago ( 2018 - 01 - 09 ) Android 4.0. x 11.1. 115.81 / September 10 , 2013 ; 4 years ago ( 2013 - 09 - 10 ) Android 2. x and 3. x 11.1. 111.73 / September 10
, 2013 ; 4 years ago ( 2013 - 09 - 10 ) Solaris 11.2. 202.223 / March 28 , 2012 ; 5 years ago ( 2012 - 03 - 28 ) Preview release ( s ) ( ± ) Windows , macOS ,
Linux , Chrome OS 28.0.
</item 5>

<item 6>
Graph query row:
work: Adobe Flash Player
platform: Windows ME
version: None
fact: None
</item 6>

<item 7>
Adobe Flash Player
152.26 , codenamed Spicy ) ( February 8 , 2011 ) Stage Video , a full hardware - accelerated video pipeline Internet Explorer 9 hardware - accelerated rendering
support Custom native mouse cursors Multiple monitor fullscreen support Enhanced subpixel rendering for text Adobe Flash Player 10.2 ( version 10.2. 152.32 ) (
February 28 , 2011 ) Adobe Flash Player 10.2 ( version 10.2. 153.1 ) ( March 21 , 2011 ) Adobe Flash Player 10.2 ( version 10.2. 159.1 ) ( April 15 , 2011 )
Adobe Flash Player 10.3 ( version 10.3.
</item 7>

<item 8>
Adobe Flash Player
181.14 , codenamed Wasabi ) ( May 12 , 2011 ) Media measurement ( video analytics for websites ; desktop only ) Acoustic Echo Cancellation ( acoustic echo
cancellation , noise suppression , voice activity detection , automatic compensation for microphone input levels ; desktop only ) Integration with browser
privacy controls for managing local storage ( ClearSiteData NPAPI ) Native control panel Auto - update notification for Mac OS X Last version for Mac OS X 10.5
and Windows 2000 ( unofficially bypassing the XP installer ) Adobe replaced Extended Support Release 10.3 by 11.7 on July 9 , 2013 . Adobe Flash Player 10.3 (
version 10.3. 181.23 ) ( June 5 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 181.26 ) ( June 14 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 181.34 ) (
June 29 , 2011 ) Adobe Flash Player 10.3 ( version 10.3.
</item 8>

<item 9>
Adobe Flash Player
183.5 ) ( August 14 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.7 ) ( August 24 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.10 ) ( September
21 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.11 ) ( November 11 , 2011 ) Adobe Flash Player 10.3 ( version 10.3. 183.25 ) ( September 18 , 2012 )
Adobe Flash Player 10.3 ( version 10.3. 183.29 ) ( October 8 , 2012 ) Adobe Flash Player 11 ( version 11.0. 1.152 , codenamed Serrano ) ( October 4 , 2011 )
Desktop only Stage 3D accelerated graphics rendering Desktop : Windows ( DirectX 9 ) , OS X ( Intel processor only ) and Linux ( OpenGL 1.3 ) , SwiftShader
fallback Mobile : Android and iOS ( OpenGL ES 2 ) H.
</item 9>

<item 10>
Graph query row:
work: Adobe Flash Player
platform: Windows 2000
version: None
fact: None
</item 10>

<item 11>
Adobe Flash Player
0.134 , codenamed Octavia ) ( March 12 , 2015 ) Control Panel improvements Installer improvements for Mac Adobe Flash Player 18 ( version 18.0. 0.160 ,
codenamed ( Presidio of San Francisco Presidio ) ) ( June 9 , 2015 ) Contains fixes for Adobe Security Bulletin APSB 15 -- 11 Adobe Flash Player 19 ( version
19.0. 0.185 , codenamed Quint ) ( September 21 , 2015 ) Adobe Flash Player 20 ( version 20.0. 0.228 , codenamed Rankin ) ( December 8 , 2015 ) Adobe Flash
Player 21 ( version 21.0. 0.182 , codenamed Sutter ) ( March 10 , 2016 ) Adobe Flash Player 22 ( version 22.0. 0.192 , codenamed Townsend ) ( June 16 , 2016 )
Adobe Flash Player 23 ( version 23.0. 0.164 , codenamed Underwood ) ( September 13 , 2016 ) Adobe Flash Player 24 ( version 24.0.
</item 11>

<item 12>
Graph query row:
work: Adobe Flash Player
platform: Mac OS X 10.6
version: None
fact: None
</item 12>

<item 13>
Adobe Flash Player
0.186 , codenamed ( Van Ness Avenue Van Ness ) ) ( December 13 , 2016 ) Adobe Flash Player 24 ( version 24.0. 0.194 ) ( January 10 , 2017 ) Adobe Flash Player
25 ( version 25.0. 0.127 , codenamed Webster ) ( March 14 , 2017 ) Adobe Flash Player 25 ( version 25.0. 0.148 ) ( April 11 , 2017 ) Adobe Flash Player 25 (
version 25.0. 0.163 ) ( April 20 , 2017 ) Adobe Flash Player 26 ( version 26.0. 0.131 , codenamed York ) ( June 13 , 2017 ) Adobe Flash Player 27 ( version
27.0. 0.130 , codenamed Zoe ) ( September 12 , 2017 ) Adobe Flash Player 27 ( version 27.0. 0.159 ) ( October 10 , 2017 ) Adobe Flash Player 28 ( version 28.0.
</item 13>

<item 14>
Adobe Flash Player
0.11 , codenamed ( Atka ) ) ( November 16 , 2017 ) See also ( edit ) Adobe AIR Adobe Shockwave Apache Flex Microsoft Silverlight References and Notes ( edit ) ^
Jump up to : `` Adobe - Flash Player '' . Adobe.com . Adobe Systems . Retrieved 9 January 2018 . ^ Jump up to : `` Archived Flash Player versions '' . Adobe.com
. Adobe Systems . Retrieved 20 October 2015 . Jump up ^ `` Download Flash Player 28 Beta '' . labs.adobe.com . Adobe Systems . Retrieved 2018 - 01 - 25 . Jump
up ^ `` Adobe Flash Player Download '' . Adobe Systems . Retrieved August 10 , 2016 . Jump up ^ `` Why You Should Ditch Adobe Shockwave '' . Krebs on Security .
May 14 , 2014 . Retrieved February 21 , 2015 . Jump up ^ `` Adobe Flash Runtimes Statistics '' . Adobe Systems Incorporated . Retrieved January 2 , 2013 .
</item 14>

<item 15>
Adobe Flash Player
Adobe Systems . November 11 , 2014 . Retrieved January 12 , 2015 . Jump up ^ `` 12 / 9 / 2014 -- Release -- Flash Player 16 '' . Adobe Systems . December 9 ,
2014 . Retrieved January 12 , 2015 . Jump up ^ `` 12 / 3 / 2015 -- Release -- Flash Player 17 '' . Adobe Systems . March 12 , 2014 . Retrieved March 13 , 2015 .
Jump up ^ `` Adobe Security Bulletin '' . Retrieved March 25 , 2017 . Further reading ( edit ) Understanding Flash Player with Adobe Scout -- an article
discussing the internals of the player and the Adobe Scout profiling tool External links ( edit ) Adobe Flash Player Flash Tester ( explains official old
working version check ) Adobe Flash File formats ActionScript ( AS ) Action Message Format ( AMF ) Flash Video ( FLV ) Flash XML Graphics ( FXG ) Local shared
objects ( LSOs ) MXML Flash Movie ( SWF ) Flash Code Library ( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark
</item 15>

<item 16>
Graph query row:
work: Adobe Flash Player
platform: Windows 8
version: None
fact: None
</item 16>

<item 17>
Graph query row:
work: Adobe Flash Player
platform: Linux
version: None
fact: None
</item 17>

<item 18>
Graph query row:
work: Adobe Flash Player
platform: Windows Server 2003
version: None
fact: None
</item 18>

<item 19>
Graph query row:
work: Adobe Flash Player
platform: Windows 98
version: None
fact: None
</item 19>
=== END ITEM id=7954a042a9de05c4 ===

=== ITEM id=738c0b6416690533 ===
Gold sentences and facts:
Sentence 0: Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 ) Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions
  Fact 0.0: Pyotr Ilyich Tchaikovsky wrote the ballet Swan Lake in 1876.
  Fact 0.1: Pyotr Ilyich Tchaikovsky wrote the ballet Sleeping Beauty in 1889.
  Fact 0.2: Pyotr Ilyich Tchaikovsky wrote the ballet The Nutcracker in 1892.
  Fact 0.3: The text includes a list of all compositions.

Retrieved items:
<item 1>
Graph query row:
composer: Pyotr Ilyich Tchaikovsky
ballet: The Nutcracker
</item 1>

<item 2>
Graph query row:
composer: Pyotr Ilyich Tchaikovsky
ballet: Swan Lake
</item 2>

<item 3>
Graph query row:
composer: Pyotr Ilyich Tchaikovsky
ballet: The Sleeping Beauty (ballet)
</item 3>

<item 4>
The Sleeping Beauty (ballet)
The work has become one of the classical repertoire 's most famous ballets . Contents ( hide ) 1 History 1.1 Composition history 1.2 Performance history 2 Roles
and original cast 3 Instrumentation 4 Roles 5 Synopsis 6 Structure 7 Versions by other hands 7.1 Piano arrangements 7.2 Aurora 's Wedding by Sergei Diaghilev 8
Trademark controversy 9 References 10 External links 10.1 Video samples 10.2 Scores 10.3 History History ( edit ) Composition history ( edit ) Tchaikovsky was
approached by the Director of the Imperial Theatres in St. Petersburg , Ivan Vsevolozhsky on 25 May 1888 about a possible ballet adaptation on the subject of
the story of Undine . It was later decided that Charles Perrault 's La Belle au bois dormant would be the story for which Tchaikovsky would compose the music
for the ballet . Tchaikovsky did not hesitate to accept the commission , although he was aware that his only previous ballet , Swan Lake , met with little
enthusiasm at that stage of his career .
</item 4>

<item 5>
The Sleeping Beauty (ballet)
With subtitles in Esperanto . Solo of Prince Désiré from Pas de deux ( Act 3 ) danced by Rudolf Nureyev ( from YouTube ) Scores ( edit ) Sleeping Beauty :
Scores at the International Music Score Library Project ( IMSLP ) History ( edit ) Tchaikovsky Research Ballets by Pyotr Ilyich Tchaikovsky Swan Lake , Op. 20
Sleeping Beauty , Op. 66 The Nutcracker , Op. 71 List of compositions by Pyotr Ilyich Tchaikovsky The ballets and * revivals of Marius Petipa in Russia 1847 --
59 Paquita ( * 1847 , * 1881 ) Le Diable amoureux ( as `` Satanella '' ) ( * 1848 ) Leda , the Swiss Milkmaid ( 1849 ) Giselle ( * 1850 , * 1884 , * 1899 , *
1903 ) The Star of Granada ( 1855 ) The Rose , the Violet , and the Butterfly ( 1857 ) Le Corsaire ( * 1858 , * 1863 , * 1868 ,
</item 5>

<item 6>
Graph query row:
composer: Tchaikovsky
ballet: Swan Lake
</item 6>

<item 7>
The Sleeping Beauty (ballet)
This version has been performed by a number of central European ballet companies over the past two decades . Roles and original cast ( edit ) Role Mariinsky
1890 Bolshoi 1899 King Florestan Feliks Krzesiński ( Mathilde Kschessinska 's father ) Queen Giuseppina Cecchetti Princess Aurora , the Sleeping Beauty Carlotta
Brianza Lyubov Roslavleva Lilac Fairy Marie Petipa M. Grachevskaya Carabosse Enrico Cecchetti Vasiliy Geltser Prince Désiré Pavel Gerdt Ivan Khlyustin Bluebird
Enrico Cecchetti Princess Florine Varvara Nikitina The ballet 's premiere received more favorable accolades than Swan Lake from the press but Tchaikovsky never
had the luxury of being able to witness his work become an instant success in theatres outside of Russia . He died in 1893 .
</item 7>

<item 8>
The Sleeping Beauty (ballet)
* 1895 ) The Sleeping Beauty ( 1890 ) Nénuphar ( 1890 ) Kalkabrino ( 1891 ) A Fairy Tale ( 1891 ) La Sylphide ( * 1892 ) The Nutcracker ( 1892 ) Cinderella (
1893 ) The Awakening of Flora ( 1894 ) Swan Lake ( * 1895 ) The Little Humpbacked Horse ( as `` The Tsar Maiden '' ) ( * 1895 ) The Calvary Halt ( 1896 ) The
Pearl ( 1896 ) Bluebeard ( 1896 ) Raymonda ( 1898 ) 1900 -- 03 Les Ruses d'Amour or The Trial of Damis ( 1900 ) The Seasons ( 1900 ) Harlequinade ( 1900 ) The
Heart of the Marquis ( 1902 ) The Magic Mirror ( 1903 ) The Romance of the Rosebud and the Butterfly ( never presented ) Pyotr Ilyich Tchaikovsky Operas The
Voyevoda Undina The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa
</item 8>

<item 9>
The Sleeping Beauty (ballet)
The ballet scenario that Tchaikovsky worked on was based on the Brothers Grimm 's version of Perrault 's work entitled ' Dornröschen ' . In that version , the
Princess 's parents ( the King and the Queen ) survived the 100 - year sleep to celebrate the Princess 's wedding to the Prince . However , Vsevolozhsky
incorporated Perrault 's other characters from his stories into the ballet , such as Puss in Boots , Little Red Riding Hood , Cinderella , Bluebird , Bluebeard
, Ricky of the Tuft and Tom Thumb . Other French fairy tale characters to be featured are Beauty and the Beast , Pretty Goldilocks and The White Cat .
Regardless , Tchaikovsky was happy to inform the Director of the Imperial Theatre that he had great pleasure studying the work and had come away with adequate
inspiration to do it justice . The choreographer was Marius Petipa , ballet master of the Imperial Ballet , who wrote a very detailed list of instructions as to
the musical requirements .
</item 9>

<item 10>
Graph query row:
composer: Tchaikovsky
ballet: The Sleeping Beauty
</item 10>

<item 11>
The Sleeping Beauty (ballet)
The Sleeping Beauty ( ballet ) - wikipedia The Sleeping Beauty ( ballet ) Jump to : navigation , search Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 )
Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions The Sleeping Beauty ( Russian : Спящая красавица / Spyashchaya krasavitsa ) is a
ballet in a prologue and three acts , first performed in 1890 . The music was composed by Pyotr Ilyich Tchaikovsky ( his opus 66 ) . The score was completed in
1889 , and is the second of his three ballets . The original scenario was conceived by Ivan Vsevolozhsky , and is based on Charles Perrault 's La Belle au bois
dormant . The choreographer of the original production was Marius Petipa . The premiere performance took place at the Mariinsky Theatre in St. Petersburg on
January 15 , 1890 . The work has become one of the classical repertoire 's most famous ballets .
</item 11>

<item 12>
The Sleeping Beauty (ballet)
The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa ( or Mazeppa ) Cherevichki The Enchantress The Queen of Spades Iolanta Ballets Swan Lake
The Sleeping Beauty The Nutcracker Symphonies No. 1 in G minor ( `` Winter Daydreams '' ) No. 2 in C minor ( `` Little Russian '' ) No. 3 in D major ( `` Polish
'' ) No. 4 in F minor Symphony in B minor ( `` Manfred '' ) No. 5 in E minor No. 6 in B minor ( `` Pathétique '' ) No. 7 in E ♭ major ( unfinished ) Concertante
Piano Concerto No. 1 in B ♭ minor Piano Concerto No. 2 in G major Piano Concerto No.
</item 12>

<item 13>
The Sleeping Beauty (ballet)
* 1896 ) 1880 -- 99 La Fille du Danube ( * 1880 ) Zoraiya ( 1881 ) La Vivandière ( as `` Markitenka '' ) ( * 1881 ) Pâquerette ( * 1882 ) The Night and the Day
( 1883 ) Pygmalion ( 1883 ) Coppélia ( * 1884 ) Giselle ( * 1884 , 1899 , 1903 ) Le Diable à Quatre ( as `` The Willful Wife '' ) ( * 1885 ) La Fille Mal Gardée
( * 1885 ) The Magic Pills ( 1886 ) The King 's Command ( 1886 , * 1887 , * 1900 ) La Esmeralda ( * 1886 , * 1899 ) The Sacrifices to Cupid ( 1886 ) * Fiametta
( 1887 ) The Vestal ( 1888 ) The Talisman ( 1889 , * 1895 ) The Enchanted Forest ( * 1889 ) The Caprices of a Butterfly ( 1889 , *
</item 13>

<item 14>
Graph query row:
composer: Pyotr Ilyich Tchaikovsky
ballet: The Sleeping Beauty
</item 14>

<item 15>
The Sleeping Beauty (ballet)
1 in D major String Quartet No. 2 in F major String Quartet No. 3 in E ♭ minor Souvenir d'un lieu cher Piano Trio in A minor String Sextet in D minor ( Souvenir
de Florence ) Piano music Scherzo à la russe Souvenir de Hapsal The Seasons ( Les saisons ) Piano Sonata No.
</item 15>

<item 16>
The Sleeping Beauty (ballet)
* 1891 , * 1903 ) Don Quixote ( 1869 , * 1871 ) Trilby ( 1870 ) Catarina ( * 1870 ) The Two Stars ( 1871 ) La Péri ( * 1872 ) Camargo ( 1872 ) Le Papillon ( *
1874 ) Ondine ( as `` The Naiad and the Fisherman '' ) ( * 1874 , * 1892 ) The Bandits ( 1875 ) The Adventures of Peleus ( 1876 , * 1897 ) A Midsummer Night 's
Dream ( 1876 ) La Bayadère ( 1877 , * 1900 ) Roxana ( 1878 ) Ariadne ( 1878 ) The Daughter of the Snows ( 1879 ) Frizak the Barber ( 1879 ) Mlada ( 1879 ,
</item 16>
=== END ITEM id=738c0b6416690533 ===

=== ITEM id=185ec28adadaa7d9 ===
Gold sentences and facts:
Sentence 0: In most of the world speed limit signs display the limitation within a red circle . This design follows the style set out by the Vienna Convention on Road Signs and Signals , with the exception of United States and Canada . All countries globally worldwide ( except the United Kingdom and United States ) use metric speed limits in kilometres per hour . Metric signs are optional in the United States though less commonly seen , while Samoa uses both units . In Ontario , Canada , the type , location , and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act .
  Fact 0.0: In most of the world, speed limit signs display the limit within a red circle.
  Fact 0.1: The red circle design follows the style set out by the Vienna Convention on Road Signs and Signals.
  Fact 0.2: The United States and Canada are exceptions to the Vienna Convention style.
  Fact 0.3: All countries worldwide, except the United Kingdom and United States, use metric speed limits in kilometres per hour.
  Fact 0.4: Metric signs are optional in the United States, though less commonly seen.
  Fact 0.5: Samoa uses both metric and imperial units for speed limits.
  Fact 0.6: In Ontario, Canada, the type, location, and frequency of speed limit signs are covered by regulation 615 of the Ontario Highway Traffic Act.

Retrieved items:
<item 1>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
In 2012 , in UK , 30 % of drivers did not comply with speed limits . In Europe , between 2009 and 2012 , 20 % of European drivers have been fined for excessive
speed . In 2012 in Europe , 62 % of people supported the idea to set up speed - limiting devices . The tolerance level needs to be adequate to not add confusion
to the driver . One efficient scheme consist in penalty points and charges for speeding just a few over the speed limit . Another possibility is to make the
necessary work to change the road so that the driver can consider the speed limit is legitimate in regard to the road . This can be achieved by implementing
traffic calming measure , vehicle activated signs , or safety cameras . The city of Munich has adopted Self - explaining roads : roadway widths , intersection
controls and crossing types have been harmonized in regard to speed limit , so that the driver can guess the speed limit even with no sign . A Self - Explaining
Road is a Self - Regulating Road . Effectiveness ( edit ) To be effective and applied by drivers , the speed limits need to be perceived as credible .
</item 1>

<item 2>
Speed limit
Maximum speed limit ( edit ) Speed limit sign for 50 km / h ( Vienna Convention Sign C14 , most of the world follows this pattern ) UK sign for 50 mph Standard
speed limit sign used in Australia showing 50 km / h Alternative Vienna Convention sign with a yellow background used in some countries , such as Sweden ,
Finland and Iceland Ireland includes the symbol `` km / h '' since going metric in 2005 Japan uses blue numerals Samoa uses both miles per hour and kilometres
per hour The United Arab Emirates uses both Western Arabic and Eastern Arabic numerals Canada ( displayed in km / h ) Canada ( Ontario ) Canada ( Yukon and
British Columbia ) United States ( in mph ) United States ( Oregon variant ) United States ( metric ) Minimum speed limit ( edit ) Minimum speed limit signs are
usually a blue circle with a white border and the limit in white . Some South American countries ( e.g. : Argentina ) , use a red border . Japan and South Korea
use their normal speed limit sign , with a line below the limit .
</item 2>

<item 3>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
, 130 km / h ( 81 mph ) freeway advisory limit , 100 km / h ( 62 mph ) on other rural roads , and 0.05 milligrams BAC ) . The Isle of Man has no speed limit on
many rural roads ; a 2004 proposal to introduce a general speed limits 60 mph and of 70 mph on Mountain Road for safety reasons were not progressed following
consultation . Measured travel speeds on the island are relatively low . Roads formerly without speed limits ( edit ) Many roads without a maximum limit became
permanently limited following the 1973 oil crisis . For example , Switzerland and Austria had no maximum restriction prior to 1973 on motorways and rural roads
, but imposed a temporary 100 km / h ( 62 mph ) maximum limit in quick response to higher fuel prices ; the limit on motorways was increased to 130 km / h ( 81
mph ) later in 1974 . Montana and Nevada were the last remaining U.S. states relying exclusively on the basic rule , without a specific , numeric rural speed
limit prior to the National Maximum Speed Law of 1974 .
</item 3>

<item 4>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
Somit konnte erstmals auf einer Autobahn die Geschwindigkeit des Verkehrs beeinflusst sowie eine Unfallwarnung vorgenommen werden . Die ersten Erfahrungen und
wissenschaftlichen Untersuchungen berichten von einer Abnahme der Störungen und Verkehrszusammenbrüche , einer Harmonisierung der Geschwindigkeitsverteilung
sowie einer Steigerung der Leistungsfähigkeit ( ZACKOR 1972 , siehe auch Kapitel 3.2. 2 ) . Jump up ^ Ralf Schmahld ( 6 August 2009 ) , ( translation ) `` 20
years waiting in traffic jams '' ; Original Title `` 20 Jahre im Stau gestanden '' , Reise ( Travel ) magazine , retrieved 2010 - 10 - 16 , ( translation ) `` A
total of 1,300 kilometers of motorways now have traffic control systems for the harmonization of traffic flow by speed limits and traffic warning , and the
government expects to expand their use : 2500 km stretch of motorway could be controlled by these dynamic control systems . ''
</item 4>

<item 5>
Speed limit
In the European Union , large signposts showing the national ( default ) speed limits of the respective country are usually erected immediately after border
crossings , with a repeater sign some 200 to 500 m ( 660 to 1,640 ft ) after the first sign . Some places provide an additional `` speed zone ahead '' ahead of
the restriction and speed limit reminder signs may appear at regular intervals which may be painted on the road surface . Signs are normally placed on both
sides of the road and in some places there are small ( less than 1 / 4 the size of the sign ) rectangular orange reflector flags attached to both upper right
corners of both signs . The speed limit sign marking the new speed zone may also have the orange flags ; this practice can be observed in New York on highways
where the speed limit varies such as New York State Route 17 . Speed limits on European Motorways in km / h Signage in many countries , especially in Europe ,
conforms to the Vienna Convention on Road Signs and Signals using black text with a red circle on a white background . In the United States , the signs are
usually rectangular with the words `` SPEED LIMIT '' and the values in black on a white background .
</item 5>

<item 6>
Speed limit
Speed limit - wikipedia Speed limit Jump to : navigation , search This article is about road speed limits . For speed of light , see special relativity . For
road speed limits in specific countries , see Speed limits by country . For rail speed limits , see Slow zone . A typical 60 km / h speed limit sign used in
Australia A typical 50 mph speed limit sign in the United States A typical 50 mph speed limit sign in the United Kingdom and 50 km / h speed limit sign in Hong
Kong Road speed limits are used in most countries to set the maximum ( or minimum in some cases ) speed at which road vehicles may legally travel on particular
stretches of road . Speed limits may be variable and in some places speed is unlimited ( e.g. on some Autobahn sections in Germany ) . Speed limits are normally
indicated on a traffic sign . Speed limits are commonly set by the legislative bodies of nations or provincial governments and enforced by national or regional
police or judicial bodies . The first maximum speed limit was the 10 mph ( 16 km / h ) limit introduced in the United Kingdom in 1861 .
</item 6>

<item 7>
Speed limit
Per distance traveled , consequences of inappropriate speed are more frequent on lower speed , lower quality roads ; in the United States , for example , the ``
speeding fatality rate for local roads is three times that for Interstates '' Citations for violations of the basic speed law without a crash have sometimes
been ruled unfairly vague or arbitrary , hence a violation the due process of law , at least in the State of Montana . Even within states , differing
jurisdictions ( counties and cities ) choose to prosecute similar cases with differing approaches . Maximum speed limits ( edit ) A speed limit sign in Ontario
In France city sign has value of speed limit A street limited at 30 km / h End of speed limit , previous speed limit applies Most public roads in most countries
have a legally assigned numerical maximum speed limit which applies on all roads unless otherwise stated ; lower speed limits are often shown on a sign at the
start of the restricted section , although the presence of streetlights or the physical arrangement of the road may sometimes also be used instead . A posted
speed limit may only apply to that road or to all roads beyond the sign that defines them depending on local laws .
</item 7>

<item 8>
Speed limit
; for example California Vehicle Code section 22350 which states that `` No person shall drive a vehicle upon a highway at a speed greater than is reasonable
... and in no event at a speed which endangers the safety of persons or property '' . The basic speed law is almost always supplemented by specific maximum or
minimum limits , but applies regardless . The reasonable speed may be different than the posted speed limit in conditions such as fog , heavy rain , ice , snow
, gravel , sharp corners , blinding glare , darkness , crossing traffic , or when there is an obstructed view of orthogonal traffic -- such as by road curvature
, parked cars , vegetation , or snow banks -- thus limiting the Assured Clear Distance Ahead ( ACDA ) . Basic speed laws are statutized reinforcements of the
centuries - old common law negligence doctrine as specifically applied to vehicular speed .
</item 8>

<item 9>
Speed limit
Variable speed limits ( edit ) Example variable speed limit sign in the United States Digital speed limit sign for variable speed limits In 1965 , the first
known experiments with variable speed limit signs took place on a 30 km stretch of German motorway A8 between Munich and the border city of Salzburg , Austria .
Mechanically variable message signs could display speeds of 60 , 80 and 100 km / h , `` danger zone '' or `` accident '' . Personnel monitored traffic using
video technology , and manually controlled the signage . Beginning in the 1970s , additional advanced traffic control systems were put into service . Modern
motorway control systems can work without human intervention using various types of sensors to measure traffic flow and weather conditions . In 2009 , 1,300 km
( 810 mi ) of German motorways were equipped with such systems . In the late 1960s , heavily traveled portions of the New Jersey Turnpike began using variable
speed limit signs , in combination with variable message signs . Officials can adjust the speed limit according to weather , traffic conditions , and
construction .
</item 9>

<item 10>
Speed limit
The U.S. Federal Highway Administration 's Manual on Uniform Traffic Control Devices provides guidelines for the appearance of speed limit signs . In Alaska and
California , speed limits are often labeled `` MAXIMUM SPEED '' instead . In Oregon , most speed limit signs just say `` SPEED '' and the number . Canada has
similar signs bearing the legend `` MAXIMUM '' and in km / h instead of mph . `` Maximum '' is used instead of `` Speed Limit '' because it has similar meanings
in English and French . Australian signs are rectangular but have a red circle like the Vienna Convention signs . The speed limit is commonly set at or below
the 85th percentile operating speed ( being the speed which no more than 15 % of traffic is exceeding ) and in the US is frequently set 4 to 8 mph ( 6 to 13 km
/ h ) below that speed . Thus , if the 85th percentile operating speed as measured by a Traffic and Engineering Survey exceeds the design speed , legal
protection is given to motorists traveling at such speeds ( design speed is `` based on conservative assumptions about driver , vehicle and roadway
characteristics '' ) .
</item 10>

<item 11>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
European studies have claimed that , whereas the effects of specific speed reduction schemes on particulate emissions from trucks are ambiguous , lower maximums
speed for trucks consistently result in lower emissions of CO and better fuel efficiency . Environmental considerations ( edit ) Speed limits can also be used
to improve local air quality issues or other factors affecting environmental quality for example the `` environmental speed limits '' in the United States
including one in an area of Texas . The European Union is also increasingly using speed limits as in response to environmental concerns . Advocacy ( edit )
Speed limits , and especially some of the methods used to attempt to enforce them , have always been controversial . There are a variety of notable
organisations and individuals who , for a variety of often passionately held views , oppose or support the use of speed limits or the way they are enforced .
Opposition ( edit ) Speed limits , and their enforcement have been opposed by various groups and for various reasons since their inception . Historically , the
AA was formed in 1905 , initially to warn members about speed traps . In more recent times some advocacy groups seek to have certain speed limits as well as
other measures removed .
</item 11>

<item 12>
Speed limit
The speeding fatality rate for local roads is three times that for Interstates Jump up ^ http://www.pe.com/articles/speed-776756-survey-limit.html Jump up ^
http://caselaw.findlaw.com/mt-supreme-court/1110919.html Jump up ^
http://vincenzeslaw.com/4-ways-virginia-reckless-driving-charges-handled-differently-fairfax-county-arlington-alexandria-stafford/ Jump up ^ `` Traffic
Operations : Road Signs '' . Government of South Australia . 2002 - 08 - 23 ( last update date ) . Retrieved 2010 - 09 - 26 . Most regulatory signs are
rectangular with a black legend on a white background ... ( exceptions include ) speed restriction signs ( symbol enclosed in a red circle ) '' Check date
values in : date = ( help ) Jump up ^ `` Speed Zoning Information '' ( PDF ) . Institute of Transportation Engineers . 22 March 2004 . Retrieved 2009 - 09 - 17
. Jump up ^ MUTCD Sections 2B.
</item 12>

<item 13>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
It states that limited evidence suggests that speed limits have a positive effect on a system wide basis . Research in 1998 showed that the reduction of some 30
mph ( 48 km / h ) United Kingdom speed limits to 20 mph ( 32 km / h ) had achieved only a 1 mph drop in speeds and no discernible reduction in accidents ; ' 20
mph speed limit zones ' which use self enforcing traffic calming achieved average speed reductions of 10 mph , child pedestrian accidents were reduced by 70 %
and child cyclist accidents by 48 % . Zones where speeds are set at 30 km / h ( or 20 mph ) are gaining popularity as they are found to be effective at reducing
crashes and increasing community cohesion . Studies undertaken in conjunction with Australia 's move from 60 km / h ( 37 mph ) speed limits to 50 km / h ( 31
mph ) in built - up areas and found that the measure was effective in reducing speed and also the frequency and severity of crashes .
</item 13>

<item 14>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
BBC News . 2008 - 12 - 10 . Jump up ^ Federal Highway Administration ( 1998 ) p. 2 Jump up ^ British Columbia Ministry of Transportation ( 2003 ) , p. v ' The
likelihood of a crash occurring is significantly greater for motorists traveling at speed slower and faster than the mean speed of traffic ' Jump up ^ ``
Synthesis of Safety Research Related to Speed and Speed Limits '' ( PDF ) . Federal Highway Administration . 1998 . Retrieved 2009 - 09 - 17 . Jump up ^
www.destatis.de ( 10 July 2013 ) . `` Unfallentwicklung auf deutschen Straßen 2012 ( Crash Development on German Roads 2012 ) '' ( PDF ) . Statistisches
Bundesamt ( Federal Statistics Office ) . Statistisches Bundesamt . Retrieved 2013 - 09 - 23 .
</item 14>

<item 15>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
A study of the impact of the replacement of 60 km / h ( 37 mph ) with 50 km / h ( 31 mph ) speed limits in New South Wales , Australia , showed only a 0.5 km /
h drop in urban areas and a 0.7 km / h drop in rural areas . The report noted that widespread community compliance would require a combination of strategies
including traffic calming treatments . A 1999 study found that the U.S. states that increased speed limits in the wake of the repeal of federally mandated speed
limits had a 15 % increase in fatalities . Information campaigns are also used by authorities to support speed limits , for example the Speeding . No one thinks
big of you . campaign in Australia 2007 . Justification ( edit ) Speed limits are set primarily to balance road traffic safety concerns with the effect on
travel time and mobility . Speed limits are also sometimes used to reduce consumption of fuel or in response to environmental concerns . Some speed limit have
also been initiated to avoid import too much gas - oil during 1973 oil crisis .
</item 15>

<item 16>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
The Association also stated that `` raising the speed limits in Denmark ( in 2004 from 110 km / h to 130 km / h ) and Italy ( 2003 increase on six - lane
highways from 130 km / h to 150 km / h ) had no negative impact on traffic safety . The number of accidental deaths even declined '' . Safe Speed , a UK
advocacy organisation campaigns for higher speed limits and to scrap speed cameras on the basis that the benefits were exaggerated and that they may actually
increase casualty levels ; their ePetition to the UK government in 2007 calling for speed cameras to be scrapped received over 25,000 signatures . Support (
edit ) Various other advocacy groups press for stricter limits and better enforcement . The Pedestrians Association was formed in the United Kingdom in 1929 to
protect the interests of the pedestrian . Their president published a critique of motoring legislation and the influence of motoring groups in 1947 title '
Murder most foul ' which laid out in an emotional but detailed way the situation as they saw it and called for tighter speed limits .
</item 16>

<item 17>
Speed limit
In international European roads , speed should be taken into account at design stage . Maximum speed recommended by international European road design standard
Road classification 60 km / h 80 km / h ( 50 mph ) 100 km / h ( 60 mph ) 120 km / h ( 75 mph ) 140 km / h ( 85 mph ) Motorway x 80 100 120 140 Express road 60
80 100 120 x Road 60 80 100 x x Minimum speed limits ( edit ) UK minimum speed limit sign Some roads also have `` minimum speed limits '' , where slow speeds
can impede traffic flow or be dangerous . Signs often use blue circles based on the obligatory signs of the Vienna Convention on Road Signs and Signals . A
Japanese minimum speed sign has the same design as a maximum speed sign , but with a horizontal line below the number . In the United States of America , they
are also identical to their respective maximum speed limit signs with the exception of the text `` MINIMUM SPEED '' . This minimum speed is not so common , as
the risk usually raise with higher speed and not with lower speed .
</item 17>

<item 18>
Speed limit
Japan and South Korea use their normal speed limit sign , with a line below the limit . Common minimum speed limit sign The United Arab Emirates uses both
Western Arabic and Eastern Arabic numerals Argentina Canada Chile Colombia Japan Philippines South Korea United States United States ( dual maximum and minimum
speeds ) United States ( metric ) United States ( metric , dual maximum and minimum speeds ) Special speed limits ( edit ) In some countries , speed limits may
apply to certain classes of vehicles , or special conditions such as night - time . Usually these speed limits will be reduced from the normal limit for safety
reasons . Australia - Speed limit during certain times Australia - Road Train speed limit New Zealand - Limited Speed Zone ( Maximum speed limit is 100 km / h ,
reduces to 50 km / h if dangerous conditions exist such as bad weather ) Romania - Trucks speed limit United States - Roadworks zone speed limit United States -
Trucks speed limit United States - Towed vehicles speed limit United States - Night time speed limit Unique speed limit sign in the United States on evacuation
routes requiring drivers to maintain the maximum safe speed Safe speed sign in South Korea Speed limit derestriction ( edit ) In some countries , derestriction
signs are used to mark where a speed zone ends .
</item 18>

<item 19>
Graph query row:
entity: Speed limits
relationship: RELATED_TO
related: Speed of light
page: Speed limit
passage: Speed limit
Road traffic safety ( edit ) See also : Road traffic safety The Pan-American Highway with central median and no freestanding obstructions increasing level of
safety at high speed According to a 2004 report from the World Health Organisation a total of 22 % of all ' injury mortality ' worldwide were from road traffic
injuries in 2002 and without ' increased efforts and new initiatives ' casualty rates would increase by 65 % between 2000 and 2020 . The report identified that
the speed of vehicles was ' at the core of the problem and said that speed limits should be set appropriately for the road function and design along with
physical measures related to the road and the vehicle and effective enforcement by the police . Road incidents are said to be the leading cause of deaths among
children 10 -- 19 years of age ( 260,000 children die a year , 10 million are injured ) . They are also occasionally set to reduce vehicle emissions or fuel use
. Maximum speed limits place an upper limit on speed choice and if obeyed can reduce the differences in vehicle speeds by drivers using the same road at the
same time .
</item 19>
=== END ITEM id=185ec28adadaa7d9 ===

=== ITEM id=534d7aa386d047bf ===
Gold sentences and facts:
Sentence 0: Dragon Ball Z ( ドラゴンボール ゼット , Doragon Bōru Zetto , commonly abbreviated as DBZ ) is the long - running anime sequel to the Dragon Ball TV series , adapted from the final twenty - six volumes of the Dragon Ball manga written by Akira Toriyama . The manga portion of the series debuted in Weekly Shōnen Jump in October 4 , 1988 and lasted until 1995 ; the anime adaptation premiered in Japan on Fuji Television on April 26 , 1989 , taking over its predecessor 's time slot , and ran until its end on January 31 , 1996 , lasting 291 episodes in Japan , and 276 episodes in the United States originally , although all 291 episodes were later broadcast when content from the first 67 episodes was restored .
  Fact 0.0: Dragon Ball Z, Japanese title Doragon Boru Zetto, is commonly abbreviated as DBZ.
  Fact 0.1: Dragon Ball Z is the long-running anime sequel to the Dragon Ball TV series.
  Fact 0.2: Dragon Ball Z is adapted from the final twenty-six volumes of the Dragon Ball manga written by Akira Toriyama.
  Fact 0.3: The Dragon Ball Z manga portion debuted in Weekly Shonen Jump on October 4, 1988.
  Fact 0.4: The Dragon Ball Z manga portion lasted until 1995.
  Fact 0.5: The Dragon Ball Z anime premiered in Japan on Fuji Television on April 26, 1989.
  Fact 0.6: The Dragon Ball Z anime took over its predecessor's time slot.
  Fact 0.7: The Dragon Ball Z anime ran until its end on January 31, 1996.
  Fact 0.8: The Dragon Ball Z anime lasted 291 episodes in Japan.
  Fact 0.9: The Dragon Ball Z anime originally lasted 276 episodes in the United States.
  Fact 0.10: All 291 Dragon Ball Z episodes were later broadcast.
  Fact 0.11: All 291 episodes were later broadcast after content from the first 67 episodes was restored.

Retrieved items:
<item 1>
List of Dragon Ball Z episodes
The series was broadcast in eight separate near - continuous blocks ( 1 - 35 , 36 - 67 , 68 - 107 , 108 - 116 , 117 - 194 , 195 - 237 , 238 - 276 , and 277 -
291 ) , with breaks varying between four months to over a year between each block . In only one instance ( between episodes 194 and 195 ) was there actually
parity between the DVD release and the actual broadcast sequence in terms of the end of one `` season '' and the beginning of the next .
</item 1>

<item 2>
Graph query row:
series: List of Dragon Ball Z episodes (season 7)
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 2>

<item 3>
Graph query row:
series: Dragon Ball Z: For Kinect
episode_count: None
episode_list: None
episode: Dragon Ball: Episode of Bardock
episode_number: None
</item 3>

<item 4>
List of Dragon Ball Z episodes
? '' ( 炎 の 決勝 ! ! 悟空 か パイクーハン か ! ? ) August 25 , 1993 September 13 , 2001 199 184 `` Do n't Let Victory Get Away ! ! Finish With an Ultra-Fast Kamehame - Ha ''
/ `` Goku vs. Pikkon '' `` Nigasu na Shōri ! ! Kimero Chōhaya Kamehameha '' ( 逃す な 勝利 ! ! 決めろ 超速 かめ はめ 波 ) September 1 , 1993 September 14 , 2001 200 185 ``
Seven Years Since Then ! Starting Today , I 'm a High School Student '' / `` Gohan Goes to High School '' `` Are kara Shichinen ! Kyō kara Boku wa Kōkōsei '' (
あれ から 7 年 !
</item 4>

<item 5>
Graph query row:
series: List of Dragon Ball Z episodes (season 2)
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 5>

<item 6>
List of Dragon Ball Z episodes
2007 6 75 -- 107 Frieza Saga Season Four February 19 , 2008 6 108 -- 139 Garlic Jr. / Trunks / Androids Saga Season Five May 27 , 2008 6 140 -- 165 Imperfect
Cell / Perfect Cell Saga Season Six September 16 , 2008 6 166 -- 194 Cell Games Saga Season Seven November 11 , 2008 6 195 -- 219 Great Saiyaman / World
Tournament Saga Season Eight February 10 , 2009 6 220 -- 253 Babidi / Majin Buu Saga Season Nine May 19 , 2009 6 254 -- 291 Fusion / Kid Buu Saga Funimation
Dragon Box Z Sets ( edit ) Name Date Discs Episodes Dragon Box Z , Vol. 1 November 17 , 2009 6 1 -- 42 Dragon Box Z , Vol. 2 February 16 , 2010 6 43 -- 84
Dragon Box Z , Vol. 3 May 4 , 2010 6 85 -- 126 Dragon Box Z , Vol.
</item 6>

<item 7>
List of Dragon Ball Z episodes
? ゼット ソード が 折れ ちゃっ た ) December 14 , 1994 October 7 , 2002 251 236 `` Birth of a Superhuman Warrior ! ! His Name is Gotenks '' / `` Gotenks is Born '' `` Gattai
Chōjin Tanjō ! ! Sono Na wa Gotenkusu '' ( 合体 超人 誕生 ! ! その 名 は ゴテンクス ) December 21 , 1994 October 8 , 2002 252 237 `` The Final Weapon is Engaged ! ? Satan Will
Save the Earth '' / `` Unlikely Friendship '' `` Saishū Heiki Shidō ! ? Satan wa Chikyū o Sukuu '' ( 最終 兵器 始動 ! ? サタン は 地球 を 救う ) January 11 , 1995 October 9 ,
2002 253 238 `` I 've Stopped Killing ! !
</item 7>

<item 8>
List of Dragon Ball Z episodes
! ブッ 飛べ ヤコン ) April 13 , 1994 October 18 , 2001 224 209 `` A Great Miscalculation ! ! Satan vs. Three Super-Warriors ! ? '' / `` Battle Supreme '' `` Daigosan !
! Satan tai Sannin no Chōsenshi ! ? '' ( 大 誤算 ! ! サタン VS3 人 の 超 戦士 ! ? ) April 20 , 1994 October 22 , 2001 225 210 `` Such Strong Kids ! ! No. 18 's Close Fight
! ? '' / `` Eighteen Unmasks '' `` Tsuyoi ze Chibikko ! ! Jūhachigō Daikusen ! ? '' ( 強い ぜ チビッコ ! ! 18 号 大 苦戦 ! ? )
</item 8>

<item 9>
Graph query row:
series: List of Dragon Ball Z episodes (season 5)
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 9>

<item 10>
Graph query row:
series: individual DVD compilations of Dragon Ball Z
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 10>

<item 11>
Graph query row:
series: Dragon Ball Z Side Story: Plan to Eradicate the Saiyans
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 11>

<item 12>
Graph query row:
series: Dragon Ball Z: Summer Vacation Special
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 12>

<item 13>
List of Dragon Ball Z episodes
( Saiyan Sized Secret ) `` Ima da Gokū ! Subete o Kaketa Saigo no Ōwaza '' ( いまだ 悟空 ! すべて を 賭け た 最後 の 大 技 ) December 20 , 1989 May 3 , 1997 August 4 , 2005 (
re-dub ) 32 24 `` Battle Power Times Ten ! ! Vegeta 's Great Metamorphosis '' / `` Vegeta ... Saiyan Style ! '' ( Spirit Bomb Away ! ) `` Sentōryoku Jūbai ! !
Bejīta Daihenshin '' ( 戦闘 力 10 倍 ! ! ベジータ 大 変身 ) January 17 , 1990 May 10 , 1997 August 8 , 2005 ( re-dub ) 33 25 `` Do n't Die , Father ! ! This is the Depth
of Gohan 's Power '' / `` Stop Vegeta Now ! ''
</item 13>

<item 14>
List of Dragon Ball Z episodes
2000 71 -- 74 ( 57 -- 60 edited ) The Summoning 20 000000002001 - 05 - 08 - 0000 May 8 , 2001 75 -- 77 ( 61 -- 63 edited ) Frieza Saga Transformation 21
000000002001 - 06 - 26 - 0000 June 26 , 2001 78 -- 81 ( 64 -- 66 edited ) Revealed 22 000000002001 - 07 - 10 - 0000 July 10 , 2001 82 -- 84 ( 67 -- 69 edited )
Death of a Prince 23 000000002001 - 08 - 21 - 0000 August 21 , 2001 85 -- 87 ( 70 -- 72 edited ) Clash 24 000000002001 - 08 - 14 - 0000 August 14 , 2001 88 --
90 ( 73 -- 75 edited ) Desperation 25 000000002001 - 09 - 04 - 0000 September 4 ,
</item 14>

<item 15>
Graph query row:
series: Dragon Ball Z: Bardock -- The Father of Goku
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 15>

<item 16>
Graph query row:
series: Dragon Ball Z Side Story: True Plan to Eradicate the Saiyans
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 16>

<item 17>
List of Dragon Ball Z episodes
2 September 18 , 2003 26 148 -- 291 + TV Special 2 Season Box Sets ( United Kingdom ) ( edit ) Name Date Discs Episodes Dragon Ball Z Season 1 July 2 , 2012 6 1
-- 39 Dragon Ball Z Season 2 August 27 , 2012 6 40 -- 74 Dragon Ball Z Season 3 October 1 , 2012 6 75 -- 107 Dragon Ball Z Season 4 November 5 , 2012 6 108 --
139 Dragon Ball Z Season 5 January 7 , 2013 6 140 -- 165 Dragon Ball Z Season 6 March 4 , 2013 6 166 -- 194 Dragon Ball Z Season 7 May 13 , 2013 6 195 -- 219
Dragon Ball Z Season 8 July 15 , 2013 6 220 -- 253 Dragon Ball Z Season 9 October 21 ,
</item 17>

<item 18>
List of Dragon Ball Z episodes
Season Episodes Japanese airdates English airdates Saiyan Saga 39 ( 1 -- 39 ) April 26 , 1989 -- March 7 , 1990 September 13 , 1996 -- September 20 , 1997 (
edited ) June 14 , 2005 -- August 18 , 2005 ( uncut ) Namek and Captain Ginyu Sagas 35 ( 40 -- 74 ) March 14 , 1990 -- January 16 , 1991 September 20 , 1997 --
April 4 , 1998 ( edited ) September 13 , 1999 -- September 20 , 1999 ( uncut ep. 68 -- 74 ) August 22 , 2005 -- October 10 , 2005 ( uncut ep. 40 -- 67 ) Frieza
Saga 33 ( 75 -- 107 ) January 23 , 1991 -- September 11 , 1991 September 21 , 1999 -- November 3 , 1999 Garlic Jr. , Trunks and Androids Sagas 32 ( 108 -- 139 )
September 18 ,
</item 18>

<item 19>
List of Dragon Ball Z episodes
2001 91 -- 93 ( 76 -- 78 edited ) Super Saiyan Goku 26 000000002001 - 10 - 16 - 0000 October 16 , 2001 94 -- 96 ( 79 -- 81 edited ) Eleventh Hour 27
000000002001 - 11 - 13 - 0000 November 13 , 2001 97 -- 100 ( 82 -- 85 edited ) Fall of a Tyrant 28 000000002001 - 11 - 13 - 0000 November 13 , 2001 101 -- 104 (
86 -- 89 edited ) Namek 's End 29 000000002001 - 12 - 11 - 0000 December 11 , 2001 105 -- 107 ( 90 -- 92 edited ) Black Water Mist 30 000000002002 - 01 - 08 -
0000 January 8 , 2002 108 -- 110 ( 93 -- 95 edited ) Garlic Jr .
</item 19>

<item 20>
Graph query row:
series: Dragon Ball Z: Bardock - The Father of Goku
episode_count: None
episode_list: None
episode: None
episode_number: None
</item 20>
=== END ITEM id=534d7aa386d047bf ===
