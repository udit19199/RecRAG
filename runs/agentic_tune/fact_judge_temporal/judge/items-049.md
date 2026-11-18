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

This file has 2 items.

=== ITEM id=785867bc8f096ffe ===
Gold sentences and facts:
Sentence 0: You can find value in player and team props by gathering information from multiple sources and tracking how players and their teammates perform.
  Fact 0.0: One can find value in player and team props by gathering information from multiple sources.
  Fact 0.1: One can find value in player and team props by tracking how players and their teammates perform.
Sentence 1: These can be team or individual player prop bets, such as which team is winning at halftime or the total number of receptions a player collects.
  Fact 1.0: Prop bets can be team prop bets.
  Fact 1.1: Prop bets can be individual player prop bets.
  Fact 1.2: A team prop bet can be which team is winning at halftime.
  Fact 1.3: An individual player prop bet can be the total number of receptions a player collects.

Retrieved items:
<item 1>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Player props could focus on how many points a player scores or how many rebounds he pulls down. These are typically provided in an over/under format.

Then, team props can range from which team will score the first bucket to the total points scored in a quarter to which team will lead at halftime. The
selection is often robust, especially when high-profile teams play.

Futures bets are exactly what their name implies: bets on events that will happen in the future. Common futures markets include:

and much more. You can usually bet on NBA futures odds throughout the year, including well before the season starts and throughout the regular season. The
further away from the event, the more likely you will find heavier odds in your favor.

Is NBA betting legal in the U.S.?

In the US, the landscape of sports betting, specifically NBA betting, has evolved considerably. While NBA betting is permitted, its legality varies across the
nation. The significant turning point came in 2018 when the US Supreme Court made a pivotal decision.

Graph facts:
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> United States
- The best movies of 2023 so far -MENTIONS-> United States
- Mattel slammed for ‘inaccurate’ Cherokee Barbie -MENTIONS-> United States
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> United States
- Israel's defense undone by reliance on technology -MENTIONS-> United States

Entities: United States, Sporting News, NBA betting, The Best NBA Betting Sites and Apps for the 2023-24 Season, US Supreme Court, 2023-24 Season, 2018 pivotal
decision, futures bets, NBA, player props, team props, sports betting
</item 1>

<item 2>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Use most of your budget on more stable bet types, such as moneylines, spreads, or totals. Focus only a small percentage of your bankroll on props.

We wouldn’t recommend using prop bets as your main bet type when building your strategy. Use most of your budget on more stable bet types, such as moneylines,
spreads, or totals. Focus only a small percentage of your bankroll on props. Exotic props are a guessing game: When large events like the Super Bowl roll
around, you’ll be inundated with exotic and entertaining prop bets. These are almost impossible to research, and pure guessing isn’t good for anyone’s bankroll
or betting strategy.

Prop bet examples for major sports

Every major sport with odds at the best sports betting apps will have prop bets. Below, we give examples of prop bets you might find when betting on major
American sports.

NBA prop bets

The NBA offers a wide range of prop bets. They vary from player to team bets and lean heavily on the statistical side of the game.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Super Bowl
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl

Entities: Super Bowl, Bankroll, Betting strategy, Exotic props, Prop Betting - What is a Prop Bet?, Sporting News, spreads, totals, NBA, prop bets, moneylines
</item 2>

<item 3>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
The NBA is one of the most popular sports to bet on in the US. NBA prop bets are an easy way to jump into basketball betting.

Proposition bets, or props, are easy to understand, numerous, and cover both player props and team props to give you plenty of variety.

Below, we look at betting NBA props at the best NBA betting sites and share some key information to make informed bets.

What are NBA props?

A prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result. In the NBA, prop bets deal with both players and teams.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> US
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> US
- Is Google Search better than the rest? And is that fair? -MENTIONS-> US
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> US
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> US

Entities: US, Sporting News, NBA, basketball betting, proposition bets, player props, team props, NBA betting sites, NBA Prop Bets - Best Team and Player Props
2023/24 Season
</item 3>

<item 4>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
You’ll often come across NBA team prop bets that are based on a team’s points, such as the first team to reach 20 points, the first team to score in a game, or
whether the total points by both teams in a contest will end up being odd or even.

Team props, also known as “game props,” don’t rely on individual players but rather on the team as a whole. Player props, on the other hand, are driven by
individual performance.

These individual props are focused on statistical milestones during a game. The most popular bets typically focus on points, rebounds, assists, and three-point
field goals. You can also bet on combination wagers, such as points and rebounds together.

In most cases, you can bet on traditional statistics for the five starters on each team.

See more:

Tips for betting on NBA props

Every bettor should develop their strategies, but we do have three tips you can use for NBA props.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, player props, NBA Prop Bets - Best Team and Player Props 2023/24 Season, team prop bets, statistical milestones, points, rebounds,
assists, three-point field goals, combination wagers, traditional statistics, NBA props
</item 4>

<item 5>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, player props, team props, NBA Prop Bets - Best Team and Player Props 2023/24 Season, usage and minutes, betting
</item 5>

<item 6>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
For the most part, you won’t come across too many exotic prop bets when using US sportsbooks, as many are just not legal bets. However, the largest sporting
events, such as the Super Bowl or March Madness, will find sportsbooks offering up a wider range of prop bets. That’s usually when you’ll find more exotic
props.

Sportsbook rules for prop bets

Before placing any kind of bet, we highly recommend you read the sportsbook’s terms and conditions. There, you will find all the rules for the various bets you
will make.

While most sportsbooks’ rules are similar, there are always small variations in how they handle certain situations that could impact your bets.

One rule that is typically used by most sportsbooks involves player props. If you bet on a player and they don’t play in the game, most sportsbooks void the bet
and refund your wager. But if that player is involved in the game even for a single play, then the bet stands.

Graph facts:
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> March Madness
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl

Entities: March Madness, Super Bowl, US, Prop Betting - What is a Prop Bet?, Sporting News, player props, prop bets
</item 6>

<item 7>
Graph query row:
sportingNewsArticle: UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl
publishedOn: 2023-12-18T22:29:26+00:00
publisher: Sporting News
articleConnections: [{'relationship': 'PUBLISHED_BY', 'entity': 'Sporting News'}, {'relationship': 'MENTIONS', 'entity': 'UTSA Roadrunners'}, {'relationship':
'MENTIONS', 'entity': 'Marshall Thundering Herd'}, {'relationship': 'MENTIONS', 'entity': "Scooter's Coffee Frisco Bowl"}, {'relationship': 'MENTIONS',
'entity': "Scooter's Coffee"}, {'relationship': 'MENTIONS', 'entity': 'West Virginia'}, {'relationship': 'MENTIONS', 'entity': "Duke's Mayo Bowl"},
{'relationship': 'MENTIONS', 'entity': 'UTSA vs. Tulane game'}, {'relationship': 'MENTIONS', 'entity': 'UTSA'}, {'relationship': 'MENTIONS', 'entity':
'Marshall'}, {'relationship': 'MENTIONS', 'entity': 'Frisco Bowl'}, {'relationship': 'MENTIONS', 'entity': 'Joshua Cephus'}, {'relationship': 'MENTIONS',
'entity': 'Jeff Traylor'}, {'relationship': 'MENTIONS', 'entity': 'Trey Moore'}, {'relationship': 'MENTIONS', 'entity': 'Syrus Dumas'}, {'relationship':
'MENTIONS', 'entity': 'Xavier Spencer'}, {'relationship': 'MENTIONS', 'entity': 'Kyle Eaves'}, {'relationship': 'MENTIONS', 'entity': 'Cam Fancher'},
{'relationship': 'MENTIONS', 'entity': 'Cole Pennington'}, {'relationship': 'MENTIONS', 'entity': 'Chad Pennington'}, {'relationship': 'MENTIONS', 'entity':
'Pennington'}, {'relationship': 'MENTIONS', 'entity': 'Rasheen Ali'}, {'relationship': 'MENTIONS', 'entity': 'Fancher'}, {'relationship': 'MENTIONS', 'entity':
'Chase Harrison'}, {'relationship': 'MENTIONS', 'entity': 'Brandon McElroy'}, {'relationship': 'MENTIONS', 'entity': 'Caleb Coombs'}, {'relationship':
'MENTIONS', 'entity': 'Zion Odoms'}, {'relationship': 'MENTIONS', 'entity': 'Jayven Hall'}, {'relationship': 'MENTIONS', 'entity': 'Thomas Lane'},
{'relationship': 'MENTIONS', 'entity': 'Chinazo Obobi'}, {'relationship': 'MENTIONS', 'entity': 'Maurice Jones'}, {'relationship': 'MENTIONS', 'entity': 'Sean
Sallis'}, {'relationship': 'MENTIONS', 'entity': 'Myles Bell'}, {'relationship': 'MENTIONS', 'entity': 'Trent Holler'}, {'relationship': 'MENTIONS', 'entity':
'Joshua McTier'}, {'relationship': 'MENTIONS', 'entity': 'Jabarrek Hopkins'}, {'relationship': 'MENTIONS', 'entity': 'CK Obobi'}, {'relationship': 'MENTIONS',
'entity': 'Sean Ellis'}, {'relationship': 'MENTIONS', 'entity': 'TyShawn Hurst'}, {'relationship': 'MENTIONS', 'entity': 'Elijah Russell'}, {'relationship':
'MENTIONS', 'entity': 'Kevorian Barnes'}, {'relationship': 'MENTIONS', 'entity': 'Robert Henry'}, {'relationship': 'MENTIONS', 'entity': 'Frank Harris'},
{'relationship': 'MENTIONS', 'entity': 'transfer portal'}, {'relationship': 'MENTIONS', 'entity': 'Tulane'}, {'relationship': 'MENTIONS', 'entity': 'Harris'},
{'relationship': 'MENTIONS', 'entity': 'ESPN'}, {'relationship': 'MENTIONS', 'entity': 'BetMGM Sportsbook'}, {'relationship': 'MENTIONS', 'entity': 'North
Carolina'}]
articlePassages: ["UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)\nThe UTSA
Roadrunners (8-4) take on the Marshall Thundering Herd (6-6) on Tuesday night in the Scooter's Coffee Frisco Bowl (9 p.m ET, ESPN). The matchup sees the
Roadrunners as large betting favorites on odds boards. Currently, USTA is favored by on the point spread, with Marshall available for on the moneyline.\n\nThe
Roadrunners are in their 13th season as a program but have yet to win a bowl game. This is the fourth bowl game they've played in five years. Marshall is
competing in a bowl game for the seventh straight year.\n\nThese teams have been on different streaks as the year has progressed. UTSA has won seven of its last
eight games, while Marshall has lost six of its last eight.\n\nHere are the live odds, team betting news, player props and our betting prediction for UTSA vs.
Marshall in Tuesday's Scooter's Coffee Frisco Bowl.", "UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News,
2023-12-18T22:29:26+00:00)\nMarshall in Tuesday's Scooter's Coffee Frisco Bowl.\n\nLive odds for UTSA vs. Marshall: Point spread, moneyline, total\n\nCheck out
UTSA vs. Marshall live odds, including the spread, moneyline and total available at the best online sportsbooks.\n\nMore: North Carolina vs. West Virginia odds
in Duke's Mayo Bowl | Best college football betting promos\n\nUTSA Roadrunners betting news\n\nUTSA has hit its stride over the second half of the season,
before losing 29-16 to Tulane in its last outing. The Roadrunners scored at least 34 points in every win since Oct. 7, proving to be an offensive juggernaut.
Everything starts with star QB Frank Harris.\n\nThe dual-threat QB has completed 64.9% of his passes this season for 2,506 yards and 18 touchdowns.", "UTSA vs.
Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)\nWhile impressive, he totaled 4,059
yards and 32 TDs on a 69.6% completion rate last year. UTSA's offense is good, but we've seen a better version of the unit in recent seasons.\n\nStill, Harris'
experience and talent often shine through. UTSA's leading receiver Joshua Cephus did the heavy lifting for the unit. He caught 82 passes for 1,049 yards and
nine touchdowns. The offense also had four ball carriers with at least 323 rushing yards, so there are plenty of threats for head coach Jeff Traylor to
unleash.\n\nUTSA averaged 31.7 points and 424.6 total yards per game, which rank 37th and 39th in the nation, respectively. Roadrunners opponents ranked 69th in
third down conversion rate, but the defense allowed an 87.5% score rate in the red zone (99th).", "UTSA vs. Marshall odds, props, predictions: Roadrunners lay
big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)\nUTSA is 5-6-1 ATS and 4-7-1 to the OVER.\n\nUTSA Roadrunners opting out of Scooter's
Coffee Frisco Bowl\n\nThe Roadrunners will be without their best defensive player, Trey Moore, as he looks to jump to a Power 5 school. Moore led the conference
in sacks, so he's a star-level absence for the unit.\n\nHere are USTA players who won't play in the Frisco Bowl.\n\nTrey Moore, EDGE (transfer portal)\n\nSyrus
Dumas, DB (transfer portal)\n\nXavier Spencer, CB (transfer portal)\n\nKyle Eaves, WR (transfer portal)\n\nMarshall Thundering Herd betting news\n\nThe toughest
blow for Marshall in this game is being without second-year starting QB Cam Fancher (transfer portal). Instead, Marshall will start freshman Cole Pennington,
who is the son of former Marshall star QB Chad Pennington.", "UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting
News, 2023-12-18T22:29:26+00:00)\nPennington has thrown six interceptions and zero touchdowns on only 79 passing attempts this season.\n\nMarshall's offense
will continue to run through star RB Rasheen Ali. The senior speedster carried the offense even when Fancher was under center, totaling 1,256 yards and 15
touchdowns on the season. This offense has struggled despite Ali's efforts, ranking 93rd in scoring and 89th in yards per game.\n\nOn the season, Marshall is
4-8 ATS and 6-6 on the O/U board. The Herd won only three of their eight conference games on the season.\n\nMarshall Thundering Herd opting out of Scooter's
Coffee Frisco Bowl\n\nWhile UTSA has the biggest individual loss for this game, the Herd's depth has taken a huge blow thanks to transfer portal entrants.",
'UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)\nChase Harrison, QB (transfer
portal)\n\nBrandon McElroy, DL (transfer portal)\n\nCaleb Coombs, WR (transfer portal)\n\nZion Odoms, WR (transfer portal)\n\nJayven Hall, LB (transfer
portal)\n\nCam Fancher, QB (transfer portal)\n\nThomas Lane, DL (transfer portal)\n\nChinazo Obobi, OT (transfer portal)\n\nMaurice Jones, RB (transfer
portal)\n\nSean Sallis, TE (transfer portal)\n\nMyles Bell, DB (transfer portal)\n\nTrent Holler, iOL (transfer portal)\n\nJoshua McTier, CB (transfer
portal)\n\nJabarrek Hopkins, DL (transfer portal)\n\nCK Obobi, OL (transfer portal)\n\nSean Ellis, TE (transfer portal)\n\nTyShawn Hurst, OT (transfer
portal)\n\nElijah Russell, EDGE (transfer portal)\n\nUTSA vs.', "UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting
News, 2023-12-18T22:29:26+00:00)\nMarshall prop bets at BetMGM Sportsbook\n\nBetMGM Sportsbook offers some excellent player props for the Scooter's Coffee
Frisco Bowl between the Roadrunners and Thundering Herd.\n\nFrank Harris prop bets\n\n237.5 passing yards OVER (-115) / UNDER (-115)\n\nPassing TDs OVER 1.5
(-125) / UNDER 1.5 (-105)\n\nAnytime touchdown scorer (-110)\n\nJoshua Cephus props bets\n\n80.5 receiving yards OVER (-115) / UNDER (-115)\n\nAnytime touchdown
scorer (-190)\n\nRasheen Ali prop bets\n\n80.5 rushing yards OVER (-115) / UNDER (-115)\n\nAnytime touchdown scorer (-175)\n\nKevorian Barnes prop bets\n\n64.5
rushing yards OVER (-115) / UNDER (-115)\n\nAnytime touchdown scorer (-135)\n\nRobert Henry prop bets\n\nAnytime touchdown scorer (-130)\n\nUTSA vs.", "UTSA vs.
Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)\nMarshall prediction: Frank Harris leads
Roadrunners past the Herd\n\nEven with the transfer portal hitting this game hard, the UTSA offense will be too much for Marshall to keep up with. The
Roadrunners push the pace in a way that Marshall can't replicate, and the Thundering Herd aren't efficient enough to make the most of what few scoring
opportunities they'll have.\n\nThe Thundering Herd could find themselves down several scores by the end of the first frame if Pennington turns the ball over, as
he's prone to do. UTSA is a well-oiled machine against teams with similar or worse talent, and Marshall is in that latter category.\n\nIt's a big number to lay,
but we're on the favorite Tuesday night.\n\nPick: UTSA"]
comparisonArticle: NBA Prop Bets - Best Team and Player Props 2023/24 Season
comparisonPublishedOn: 2023-10-02T23:49:35+00:00
comparisonConnections: [{'relationship': 'PUBLISHED_BY', 'entity': 'Sporting News'}, {'relationship': 'MENTIONS', 'entity': '2023/24 season'}, {'relationship':
'MENTIONS', 'entity': 'US'}, {'relationship': 'MENTIONS', 'entity': 'NBA'}, {'relationship': 'MENTIONS', 'entity': 'basketball betting'}, {'relationship':
'MENTIONS', 'entity': 'proposition bets'}, {'relationship': 'MENTIONS', 'entity': 'player props'}, {'relationship': 'MENTIONS', 'entity': 'team props'},
{'relationship': 'MENTIONS', 'entity': 'NBA betting sites'}, {'relationship': 'MENTIONS', 'entity': 'prop bets'}, {'relationship': 'MENTIONS', 'entity': 'sports
betting'}, {'relationship': 'MENTIONS', 'entity': 'sportsbook'}, {'relationship': 'MENTIONS', 'entity': 'NBA game'}, {'relationship': 'MENTIONS', 'entity': 'NBA
prop bets'}, {'relationship': 'MENTIONS', 'entity': 'Team and player props'}, {'relationship': 'MENTIONS', 'entity': 'online sports betting'}, {'relationship':
'MENTIONS', 'entity': 'online sportsbook'}, {'relationship': 'MENTIONS', 'entity': 'terms and conditions'}, {'relationship': 'MENTIONS', 'entity': 'welcome
bonus'}, {'relationship': 'MENTIONS', 'entity': 'deposit'}, {'relationship': 'MENTIONS', 'entity': 'e-wallet'}, {'relationship': 'MENTIONS', 'entity':
'basketball'}, {'relationship': 'MENTIONS', 'entity': 'National Basketball Association'}, {'relationship': 'MENTIONS', 'entity': 'team prop bets'},
{'relationship': 'MENTIONS', 'entity': 'statistical milestones'}, {'relationship': 'MENTIONS', 'entity': 'points'}, {'relationship': 'MENTIONS', 'entity':
'rebounds'}, {'relationship': 'MENTIONS', 'entity': 'assists'}, {'relationship': 'MENTIONS', 'entity': 'three-point field goals'}, {'relationship': 'MENTIONS',
'entity': 'combination wagers'}, {'relationship': 'MENTIONS', 'entity': 'traditional statistics'}, {'relationship': 'MENTIONS', 'entity': 'NBA props'},
{'relationship': 'MENTIONS', 'entity': 'usage and minutes'}, {'relationship': 'MENTIONS', 'entity': 'betting'}, {'relationship': 'MENTIONS', 'entity': 'Giannis
Antetokounmpo'}, {'relationship': 'MENTIONS', 'entity': 'LeBron James'}]
comparisonPassages: ['NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nThe NBA is one of the most popular
sports to bet on in the US. NBA prop bets are an easy way to jump into basketball betting.\n\nProposition bets, or props, are easy to understand, numerous, and
cover both player props and team props to give you plenty of variety.\n\nBelow, we look at betting NBA props at the best NBA betting sites and share some key
information to make informed bets.\n\nWhat are NBA props?\n\nA prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result.
In the NBA, prop bets deal with both players and teams.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News,
2023-10-02T23:49:35+00:00)\nIn the NBA, prop bets deal with both players and teams.\n\nSome examples of NBA player props are:\n\nHow many points a player will
score\n\nNumber of rebounds a player will have\n\nHow many assists a player will get\n\nRebounds and assists combined\n\nPoints, rebounds, and assists
combined\n\nHow many steals a player will get\n\nNumber of three-pointers made\n\nSome examples of NBA team props include:\n\nWhich team will reach 10 points
first\n\nWhich team will have the highest-scoring period\n\nWhich team will score first\n\nWhich team will win the first half\n\nWhich team will win the second
half\n\nWill there be a fourth-quarter comeback victory\n\nAll of the best sports betting sites have their selection of prop bets for each NBA game, so shop
around to find the sportsbooks that best suit your needs.\n\nHow to bet on NBA props\n\nThe first step to betting on NBA props is opening an account at an
online sportsbook. The process is simple, free, and doesn’t take very long.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News,
2023-10-02T23:49:35+00:00)\nThe process is simple, free, and doesn’t take very long. Below are the steps to opening a new account at an online
sportsbook.\n\nChoose your sportsbook: Any state with legal online sports betting has a variety of sportsbooks. If you aren’t sure which sportsbooks best fit
your needs, research and read our reviews on each. And don’t forget that you can have an account with more than one. Once you decide which sportsbook you’re
interested in, follow the link from our site to qualify for the best new customer welcome bonus.\n\nRegister for an account: At the top of the homepage of the
sportsbook website, you will find a button that says “register” or “sign up.” Click that, and you’ll be taken to a form you must fill out. It will ask for
personal information such as your name, phone number, email address, last four digits of your Social Security number, and more.', 'NBA Prop Bets - Best Team and
Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nAccept the terms and conditions: Before finalizing your account information, you must
accept the sportsbook’s terms and conditions. We recommend you read through these to understand the rules your sportsbook will enforce regarding bets,
especially for the sports you intend to wager on. The terms and conditions are also available on the sportsbook website for you to read at any point.\n\nGet
your bonus: At this point, you can accept your welcome bonus. Be sure to read through the bonus requirements before accepting anything. If a bonus code is
needed, which you can find on our website, this is where you will input it during the registration process.\n\nMake a deposit: Now that you’ve signed up for an
account, you can start funding your bankroll by making a deposit. Most legal sportsbooks have a variety of ways you can add funds to your account, such as
through your bank or an e-wallet.\n\nStart betting: Go to the basketball section of your sportsbook and select the NBA.', 'NBA Prop Bets - Best Team and Player
Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nStart betting: Go to the basketball section of your sportsbook and select the NBA. From there,
you can choose individual games and make bets like money lines, point spreads, totals, props, futures, and more.\n\nHow to read NBA props odds\n\nAt first
glance, all the positive and negative numbers at US sportsbooks can be daunting. The good news is they are easy to read once you understand what you
see.\n\nNegative numbers, such as -110, represent how much money you need to wager to win $100. In many cases, such as a moneyline bet, the negative number also
indicates which side of the bet is the favorite to win.\n\nFor example, if you were to bet on LeBron James scoring over 15.5 points in a contest at -125 odds,
you’d need to wager $125 to win $100. If you did and your bet was correct, then you’d take home $225, which is your original $125 stake returned to you plus
your $100 in profit.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nPositive numbers, like +145, show
how much you stand to win for every $100 you wager. If you were to bet $50 on Giannis Antetokounmpo to have over 7.5 rebounds and the odds were +130 at the time
you finalized your bet slip, you would win $65 in profit if your bet wins.\n\nPositive numbers also indicate which side of the bet is the underdog in the eyes
of the oddsmakers.\n\nRemember that you can bet as little or as much as you want, and your payout will be calculated based on your wager amount and the odds at
the time you make your bet.\n\nTypes of NBA props\n\nYou can make two main prop bets on the National Basketball Association:\n\nTeam props\n\nPlayer
props\n\nTeam props are typically focused on statistical accomplishments by a particular team and rarely on the game’s result.', 'NBA Prop Bets - Best Team and
Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nYou’ll often come across NBA team prop bets that are based on a team’s points, such as
the first team to reach 20 points, the first team to score in a game, or whether the total points by both teams in a contest will end up being odd or
even.\n\nTeam props, also known as “game props,” don’t rely on individual players but rather on the team as a whole. Player props, on the other hand, are driven
by individual performance.\n\nThese individual props are focused on statistical milestones during a game. The most popular bets typically focus on points,
rebounds, assists, and three-point field goals. You can also bet on combination wagers, such as points and rebounds together.\n\nIn most cases, you can bet on
traditional statistics for the five starters on each team.\n\nSee more:\n\nTips for betting on NBA props\n\nEvery bettor should develop their strategies, but we
do have three tips you can use for NBA props.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nPay
attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.\n\nKeep an eye on usage and minutes: Even
slight increases to a player’s average minutes can make a difference when betting player props, especially later in the season when aging veteran players tend
to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use them to your advantage.', 'NBA Prop Bets
- Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nRemember that some of the best players in the NBA were once on the bench
for most games until small increases in their playing time gave them a chance to shine.\n\nShop for the best lines: This tip applies to every bet you make.
Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the smallest differences, like one sportsbook having odds at +125
and another at +135, can make a difference in your profit. By shopping for the best lines, you can make smarter bets and find the best value.\n\nRelated
Pages:']
</item 7>

<item 8>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
They vary from player to team bets and lean heavily on the statistical side of the game. Some NBA prop examples include:

How many points a player will score

Number of assists or three-pointers a player will have

Which player will score the first basket of a game

You can also bet on team props, such as whether a team will lead at the end of a certain quarter or an over/under bet on the total number of points the team
will score in the game.

See more: NBA prop bets

NFL prop bets

The National Football League is the king of sports betting, and NFL betting sites have no qualms about showcasing every kind of possible bet type for it.

Bettors will find many prop bet opportunities, especially for the Super Bowl. During the regular season, one of the most popular prop bet options includes the
“anytime touchdown scorer” bet, where you can wager on a player to score a TD at any point in the game. Other prop bets can be focused on the team or even both
teams.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Super Bowl
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl

Entities: Super Bowl, prop betting, Prop Betting - What is a Prop Bet?, Sporting News, NBA, sports betting, National Football League
</item 8>

<item 9>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
In the NBA, prop bets deal with both players and teams.

Some examples of NBA player props are:

How many points a player will score

Number of rebounds a player will have

How many assists a player will get

Rebounds and assists combined

Points, rebounds, and assists combined

How many steals a player will get

Number of three-pointers made

Some examples of NBA team props include:

Which team will reach 10 points first

Which team will have the highest-scoring period

Which team will score first

Which team will win the first half

Which team will win the second half

Will there be a fourth-quarter comeback victory

All of the best sports betting sites have their selection of prop bets for each NBA game, so shop around to find the sportsbooks that best suit your needs.

How to bet on NBA props

The first step to betting on NBA props is opening an account at an online sportsbook. The process is simple, free, and doesn’t take very long.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, prop bets, sports betting, sportsbook, NBA game, NBA Prop Bets - Best Team and Player Props 2023/24 Season
</item 9>

<item 10>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Remember that some of the best players in the NBA were once on the bench for most games until small increases in their playing time gave them a chance to shine.

Shop for the best lines: This tip applies to every bet you make. Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the
smallest differences, like one sportsbook having odds at +125 and another at +135, can make a difference in your profit. By shopping for the best lines, you can
make smarter bets and find the best value.

Related Pages:

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, NBA prop bets, NBA Prop Bets - Best Team and Player Props 2023/24 Season
</item 10>
=== END ITEM id=785867bc8f096ffe ===

=== ITEM id=65ead423605d9631 ===
Gold sentences and facts:
Sentence 0: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 0.0: The roundup's writers learned in the last roundup that Google spent $26.3 billion in 2021 making Google the default search engine across platforms.
  Fact 0.1: The roundup's writers learned in the last roundup that Google tried to have Chrome preinstalled on iPhones.
Sentence 1: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 1.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 1.1: Prabhakar Raghavan revealed in his court testimony that Google paid $26.3 billion in 2021 to multiple browsers, phones and platforms.
  Fact 1.2: The payments were made to companies including Apple, Samsung and Mozilla.
  Fact 1.3: The Verge reports this.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that this siphoning happens through anticompetitive means.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Mozilla, Apple, TechCrunch, Prabhakar Raghavan, Pichai, trial hearing, court testimony, Google antitrust case, decision made with the consumer
in mind, 5 things we learned so far about the Google antitrust case, Samsung, Google, New York Times, iPhones
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bellshaw
- Microsoft -MENTIONS-> cloud gaming
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft

Entities: Bellshaw, Microsoft, TechCrunch, Pichai, Google antitrust case, message exchange, litigation hold, personnel matter, Internet Explorer, David
Drummond, Internet Explorer became the default search engine, Drummond’s letter, default settings, 5 things we learned so far about the Google antitrust case,
Google
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Expedia, Jeff Hurst, The Seattle
Times, search engine quality, European antitrust fine, European Union antitrust order, Yahoo, Firefox, Android, antitrust, TechCrunch, DOJ, European Union
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Kent Walker, trial on Google, technology industry, Is Google Search better than the rest? And is that fair?, The Age, Google
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Meagan Bellshaw
- U.S. Justice Department -PARTICIPATED_IN-> U.S. Justice Department lawsuit against Google
- Meagan Bellshaw -WORKS_FOR-> Justice Department
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. Justice Department
- U.S. Justice Department -PARTICIPATED_IN-> 2020 Justice Department civil antitrust suit over search and search advertising

Entities: Meagan Bellshaw, U.S. Justice Department, TechCrunch, Pichai, Nitin, homepage market share, search market share, Google antitrust case, 5 things we
learned so far about the Google antitrust case, Google, The Verge
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Now, onto the roundup!

A window into Google’s most popular search queries

Judge Amit Mehta ruled to make a list public that provides a glimpse of which search terms make Google the most money. The list of popular search terms ordered
by revenue includes 20 terms that were lucrative for the week of September 22, 2018. Information like revenue per search term, how many queries each of those
terms got, along with a separate list of popular search terms ordered by queries (not revenue), were all redacted. The list we can see is as follows:

iPhone 8

iPhone 8 plus

Auto insurance

Car insurance

Cheap flights

Car insurance quotes

DirecTV

Online colleges

AT&T

Hulu

iPhone

Uber

Spectrum

Comcast

Xfinity

Insurance quotes

Free credit report

Cheap car insurance

AARP

LifeLock

There is, in reality, little surprise here.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- Amit Mehta -SAID-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit

Entities: Amit Mehta, Vertical search, Pre-installs, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google
trial, iPhone 8 plus, Auto insurance, Car insurance, Cheap flights, Car insurance quotes, DirecTV, Online colleges, AT&T, Spectrum, Comcast
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Chrome, Expedia, Yahoo, Firefox,
Android OEMs, antitrust, Microsoft, TechCrunch, Google, Justice Department, iPhones, U.S.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Baker, search technology,
Mozilla, Firefox, TechCrunch, Google, Bloomberg
</item 8>

<item 9>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, The Age, Google
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Satya Nadella, DuckDuckGo, Bing, The Information, Microsoft, Apple, TechCrunch, Tim Cook, Pichai, Google antitrust case, 5 things we learned
so far about the Google antitrust case, Google, iOS, iPhone
</item 10>
=== END ITEM id=65ead423605d9631 ===
