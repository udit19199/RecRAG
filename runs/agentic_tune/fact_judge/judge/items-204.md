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

This file has 3 items.

=== ITEM id=7bc805783aa8bd37 ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: That allows sportsbooks to earn the profit built into the odds and lines without worrying about which side wins.
  Fact 1.0: This allows sportsbooks to earn the profit built into the odds and lines.
  Fact 1.1: Sportsbooks earn that profit without worrying about which side wins.

Retrieved items:
<item 1>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 1>

<item 2>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.
</item 2>

<item 3>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 3>

<item 4>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Favorites to win the game are shown negative odds, such as -135, while the underdogs will have positive odds, like +165.

NBA over/under bets: Also known as totals. These bets are focused on the combined point total of the two teams playing in a game. (Winners and losers don’t
matter.) For example, the over/under could be set at 211.5 points in the Knicks vs. Heat matchup. If you were to pick the over on the bet, you’d need the two
teams to combine for 212 points or more to win. They need to score 211 points or fewer if you went with the under.

NBA props and futures

NBA player and team props are growing in popularity. Sportsbooks are taking note and providing more NBA props than ever.

After all, it’s much easier to focus on a single player and how he will perform than the entire team. Player props could focus on how many points a player
scores or how many rebounds he pulls down.
</item 4>

<item 5>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, you’d have a winner at the fourth. If you didn’t shop around, you’d never have found that half-point victory and be stuck with a push instead of a
profit.

What is the Vegas line?

Las Vegas sportsbooks set the standard for lines and odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line. Most
online betting sites don’t stray too far from the Vegas line, using it as a foundation for setting all lines in every major sport.

If those lines do shift, it’s usually because of increased betting on one side of a bet. Shop lines locally, and you can take advantage when sportsbooks stray
from the Vegas line.

Why do the lines move?

It’s worth reiterating that sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they earn the small profit built into
the odds without concern about who wins or loses.
</item 5>

<item 6>
Graph query row:
article: NBA Rookie of the Year Odds - Wembanyama the early favorite
connected_entities: [{'entity': 'Sporting News', 'relationship': 'PUBLISHED_BY'}, {'entity': 'NFL', 'relationship': 'MENTIONS'}, {'entity': 'Victor Wembanyama',
'relationship': 'MENTIONS'}, {'entity': 'Gregg Popovich', 'relationship': 'MENTIONS'}, {'entity': 'Spurs', 'relationship': 'MENTIONS'}, {'entity': 'NBA Rookie
of the Year award', 'relationship': 'MENTIONS'}, {'entity': 'NBA futures betting', 'relationship': 'MENTIONS'}, {'entity': 'NBA Rookie of the Year odds',
'relationship': 'MENTIONS'}, {'entity': 'Wembanyama', 'relationship': 'MENTIONS'}, {'entity': 'OKC', 'relationship': 'MENTIONS'}, {'entity': 'NBA Rookie of the
Year', 'relationship': 'MENTIONS'}, {'entity': 'NBA Rookie of the Year betting', 'relationship': 'MENTIONS'}, {'entity': 'Wilt Chamberlain Trophy',
'relationship': 'MENTIONS'}, {'entity': 'Paolo Banchero', 'relationship': 'MENTIONS'}, {'entity': 'Orlando Magic', 'relationship': 'MENTIONS'}, {'entity':
'Scottie Barnes', 'relationship': 'MENTIONS'}, {'entity': 'Toronto Raptors', 'relationship': 'MENTIONS'}, {'entity': 'LaMelo Ball', 'relationship': 'MENTIONS'},
{'entity': 'Charlotte Hornets', 'relationship': 'MENTIONS'}, {'entity': 'Ja Morant', 'relationship': 'MENTIONS'}, {'entity': 'Memphis Grizzlies',
'relationship': 'MENTIONS'}, {'entity': 'Luka Dončić', 'relationship': 'MENTIONS'}, {'entity': 'Dallas Mavericks', 'relationship': 'MENTIONS'}, {'entity': 'Ben
Simmons', 'relationship': 'MENTIONS'}, {'entity': 'Philadelphia 76ers', 'relationship': 'MENTIONS'}, {'entity': 'Malcolm Brogdon', 'relationship': 'MENTIONS'},
{'entity': 'Milwaukee Bucks', 'relationship': 'MENTIONS'}, {'entity': 'Karl-Anthony Towns', 'relationship': 'MENTIONS'}, {'entity': 'Minnesota Timberwolves',
'relationship': 'MENTIONS'}, {'entity': 'Andrew Wiggins', 'relationship': 'MENTIONS'}, {'entity': 'Michael Carter-Williams', 'relationship': 'MENTIONS'},
{'entity': 'Damian Lillard', 'relationship': 'MENTIONS'}, {'entity': 'Portland Trail Blazers', 'relationship': 'MENTIONS'}, {'entity': 'Kyrie Irving',
'relationship': 'MENTIONS'}, {'entity': 'Cleveland Cavaliers', 'relationship': 'MENTIONS'}, {'entity': 'Blake Griffin', 'relationship': 'MENTIONS'}, {'entity':
'Los Angeles Clippers', 'relationship': 'MENTIONS'}, {'entity': 'Tyreke Evans', 'relationship': 'MENTIONS'}, {'entity': 'Sacramento Kings', 'relationship':
'MENTIONS'}, {'entity': 'Caesars', 'relationship': 'MENTIONS'}, {'entity': 'BetMGM', 'relationship': 'MENTIONS'}, {'entity': 'US', 'relationship': 'MENTIONS'},
{'entity': 'Rookie of the Year', 'relationship': 'MENTIONS'}, {'entity': 'NBA MVP', 'relationship': 'MENTIONS'}, {'entity': 'DraftKings', 'relationship':
'MENTIONS'}, {'entity': 'FanDuel', 'relationship': 'MENTIONS'}, {'entity': 'BetMGM Sportsbook', 'relationship': 'MENTIONS'}, {'entity': 'Caesars Sportsbook',
'relationship': 'MENTIONS'}, {'entity': 'NBA', 'relationship': 'MENTIONS'}, {'entity': 'sports betting', 'relationship': 'MENTIONS'}, {'entity': 'online
sportsbook', 'relationship': 'MENTIONS'}, {'entity': 'National Basketball Association', 'relationship': 'MENTIONS'}, {'entity': 'betting odds', 'relationship':
'MENTIONS'}, {'entity': 'Paolo Banchero', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Scottie Barnes', 'relationship': 'PARTICIPATED_IN'}, {'entity':
'LaMelo Ball', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Ja Morant', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Luka Dončić', 'relationship':
'PARTICIPATED_IN'}, {'entity': 'Ben Simmons', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Malcolm Brogdon', 'relationship': 'PARTICIPATED_IN'}, {'entity':
'Karl-Anthony Towns', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Andrew Wiggins', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Michael Carter-Williams',
'relationship': 'PARTICIPATED_IN'}, {'entity': 'Damian Lillard', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Kyrie Irving', 'relationship':
'PARTICIPATED_IN'}, {'entity': 'Blake Griffin', 'relationship': 'PARTICIPATED_IN'}, {'entity': 'Tyreke Evans', 'relationship': 'PARTICIPATED_IN'}]
passages: ["NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)\nBettors across the country can wager on
which player they believe will win the coveted NBA Rookie of the Year award as part of NBA futures betting.\n\nLike other NBA futures odds, this can be bet on
before the start of the NBA regular season and throughout the year. The best NBA betting apps make it simple and convenient.\n\nBelow, you can check the latest
NBA Rookie of the Year odds as posted at online sportsbooks and how to bet on them.\n\nCurrent NBA ROTY odds\n\nCheck the current NBA ROTY odds below. Click on
any odds to claim your bonus and get started.\n\nWho are the NBA RoY odds favorites?\n\nVictor Wembanyama: Coach Gregg Popovich's Spurs have a gem in Victor
Wembanyama, a generational talent and a strong Rookie of the Year contender. Even with a young roster and potential load management, Wembanyama's undeniable
impact is evident.\n\nHis elite defense, highlighted by his 7-foot-4 height and 8-foot wingspan, allows him to dominate where others can't.", "NBA Rookie of the
Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)\nHis wingspan and defensive prowess will challenge opponents. Offensively,
he capitalizes on mismatches and has diverse scoring abilities, from the paint to the 3-point line. His underrated passing syncs well with OKC's pace and
style.\n\nHow to bet on NBA Rookie of the Year\n\nThe first thing you need when preparing to bet on the NBA Rookie of the Year Award, or any other bet, is an
account at an online sportsbook.\n\nSigning up for a new betting account doesn’t take much time. Once you visit the online sportsbook you’re interested in,
click a “sign up” or “register” button. This will take you to a form for your details. Fill that out and accept the sportsbook’s terms and conditions.\n\nAt
that point, you’ll be able to claim your welcome bonus, which you’ll qualify for by following the link on this page from our site to the online sportsbook. If
you need to enter a code to get the bonus, you’ll also find that code on our site.", 'NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting
News, 2023-10-04T22:54:23+00:00)\nWe urge you to read the requirements for any available promotions and offers before accepting them so you’re aware of any
restrictions before betting.\n\nYou can now place your initial deposit, which is easy at online betting sites thanks to various deposit and withdrawal
options.\n\nTo bet on the NBA Rookie of the Year, navigate to the basketball section of the sportsbook website, select the NBA, and go to the futures
tab.\n\nYou’ll find Rookie of the Year odds, NBA MVP odds, and more there. Select the bet you want to make, enter the amount you want to bet, and finalize your
bet slip.\n\nEnsure you’re using only legal online sportsbooks for any type of betting. Offshore sportsbooks advertise widely on the internet, but they are
illegal and offer no protection against fraud and other criminal activities.\n\nBest sportsbooks for NBA Rookie of the Year odds\n\nHere are some of the best
sports betting sites where you can bet on the next NBA Rookie of the Year.', 'NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News,
2023-10-04T22:54:23+00:00)\nDraftKings Sportsbook: DraftKings, known as a daily fantasy sports operator before becoming a renowned sportsbook, offers up a nice
variety of futures bets, including the NBA Rookie of the Year award. The online sportsbook and betting app are intuitive and offer an easy-to-understand betting
experience. Additionally, DraftKings fields a competitive welcome bonus for new players.\n\nDraftKings, known as a daily fantasy sports operator before becoming
a renowned sportsbook, offers up a nice variety of futures bets, including the NBA Rookie of the Year award. The online sportsbook and betting app are intuitive
and offer an easy-to-understand betting experience. Additionally, DraftKings fields a competitive welcome bonus for new players. FanDuel Sportsbook: Also known
for being in the DFS market, FanDuel has one of the largest followings in the country. It offers competitive odds for NBA ROTY and has one of the best sports
betting apps. Look for a quality welcome bonus for new players and attractive promotions for existing customers.', 'NBA Rookie of the Year Odds - Wembanyama the
early favorite (Sporting News, 2023-10-04T22:54:23+00:00)\nLook for a quality welcome bonus for new players and attractive promotions for existing
customers.\n\nAlso known for being in the DFS market, FanDuel has one of the largest followings in the country. It offers competitive odds for NBA ROTY and has
one of the best sports betting apps. Look for a quality welcome bonus for new players and attractive promotions for existing customers. BetMGM Sportsbook: As
one of the most recognizable names in the gambling industry, BetMGM knows how to attract and keep customers with competitive odds for all bet types, including
futures bets and the NBA Rookie of the Year. BetMGM offers many deposit and withdrawal options and 24/7 customer service, and generous sports betting bonuses
and promotions.\n\nAs one of the most recognizable names in the gambling industry, BetMGM knows how to attract and keep customers with competitive odds for all
bet types, including futures bets and the NBA Rookie of the Year. BetMGM offers many deposit and withdrawal options and 24/7 customer service, and generous
sports betting bonuses and promotions.', 'NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)\nCaesars
Sportsbook: Caesars knows its way around sports betting when you use its app and online sportsbook. It offers favorable odds for almost every bet type,
including NBA ROTY betting, and offers a nice variety of deposit and withdrawal options. Additionally, Caesars has a competitive welcome bonus for new players
and runs NBA betting promos for existing players.\n\nWhen is the best time for NBA ROTY odds?\n\nThe earlier you bet on NBA ROTY, the better the odds. But
you’ll be doing so without much information that could be key in making an educated decision.\n\nConversely, if you wait for the data you need for an informed
bet, the odds will tighten up, and you will see less of a return on your wager.\n\nWhen the lines are first released for NBA ROTY honors, the season hasn’t even
started yet, so there are no statistics, trends, or player news. The pre-season odds are so favorable because even the oddsmakers don’t have the data to craft
more specific lines.', "NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)\nOnce the season has started and
you get to see how the various rookies are performing, you can gather the relevant data. But sportsbooks are also gathering this information, and they begin to
tighten the NBA Rookie of the Year lines in response.\n\nSo, while you can better guess which player will hoist the Wilt Chamberlain Trophy, the odds will be
tighter, and your potential return on your wager will be reduced.\n\nPros and cons of betting early on the NBA Rookie of the Year:\n\nPros:\n\nYou have an
opportunity to find better value.\n\nThe betting odds are fresh and haven't been adjusted.\n\nCons:\n\nThere is a much higher risk as a player's performance,
health, and team dynamic can change throughout the season.\n\nYou’ll find limited information to make an informed bet.\n\nPros and cons of waiting to bet on the
NBA Rookie of the Year:\n\nPros:\n\nThere is a lot more information available to make an informed bet.\n\nYou don’t risk betting on a player not remotely in
contention.\n\nCons:\n\nThere is a reduced opportunity to find value.", 'NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News,
2023-10-04T22:54:23+00:00)\nCons:\n\nThere is a reduced opportunity to find value.\n\nThe odds may not be as favorable.\n\nKeep in mind that it’s always
important to take into consideration your risk tolerance while you weigh the potential rewards. No matter when you decide to make your wager, always gather as
much information as possible before risking your hard-earned money.\n\nWho are the past NBA ROTY winners?\n\nFor context, here is a list of the National
Basketball Association Rookie of the Year winners since 2010.', 'NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News,
2023-10-04T22:54:23+00:00)\nSeason Player Position Team 2023 Paolo Banchero Forward Orlando Magic 2022 Scottie Barnes Forward Toronto Raptors 2021 LaMelo Ball
Guard Charlotte Hornets 2020 Ja Morant Guard Memphis Grizzlies 2019 Luka Dončić Guard/Forward Dallas Mavericks 2018 Ben Simmons Forward/Guard Philadelphia 76ers
2017 Malcolm Brogdon Guard Milwaukee Bucks 2016 Karl-Anthony Towns Center Minnesota Timberwolves 2015 Andrew Wiggins Forward/Guard Minnesota Timberwolves 2014
Michael Carter-Williams Guard Philadelphia 76ers 2013 Damian Lillard Guard Portland Trail Blazers 2012 Kyrie Irving Guard Cleveland Cavaliers 2011 Blake Griffin
Forward Los Angeles Clippers 2010 Tyreke Evans Guard/Forward Sacramento Kings\n\nWhat other NBA futures can I bet on?\n\nRookie of the Year is not the only form
of NBA futures betting.', 'NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)\nRookie of the Year is not the
only form of NBA futures betting. Legal online sportsbooks have a wide selection of NBA futures you can bet on, including:\n\nIn addition to futures bets, you
can wager on many NBA bets during the regular season and postseason. NBA game lines (moneylines, point spreads, and totals) are the most popular, but you’ll
also be able to bet on NBA team and player props, parlays, and more.\n\nThe NBA is the second-most popular odds market to bet on in the US, behind NFL betting
odds. That means every online betting site always has a great selection of NBA odds.']
</item 6>

<item 7>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks want even betting so much that they’ll move a line or change the odds to draw more bets to the other side to get it that way.

That’s when you’ll see a difference between the lines and odds. Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like
an injury or roster move. It’ll be difficult for you to find value in that case.

But when a line moves or odds change because of betting patterns, only the sportsbook that saw the betting makes the change. That’s your opportunity right
there.

Line shopping is a tool to help you spot changes due to betting patterns. Then, you can simply take advantage of movement in a line or changes in the odds.

Choose a side, shop around at online sportsbooks, compare the lines and odds, and then pounce when you find one that has something different and more favorable
to your viewpoint.

See more: What is a parlay bet?
</item 7>

<item 8>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Either way, the line shopping strategy is designed to help you maximize your potential return.

NBA live betting

Thanks to the increased popularity of online sports betting apps, in-game betting has exploded nationwide. Also known as live betting, in-game betting allows
you to bet on a game while it’s unfolding.

The action is fast, and the odds are constantly shifting, making for a new way to enjoy the flow of a game while still staying engaged as a bettor.

Depending on the sportsbook, there is usually a strong variety of live betting opportunities. You’ll find game and player prop bets like whether the next bucket
will be worth one, two, or three points.

You can bet on the team leading at the end of the current quarter or on whether there will be a comeback victory. You can even boil live betting down to the
results of the next possession.

The most popular sportsbook apps, such as DraftKings, FanDuel, and BetMGM, offer NBA in-game betting, though each has its lines and betting opportunities.
</item 8>

<item 9>
Graph query row:
article: Line Shopping in Sports Betting - How to Find the Best Odds
connected_entities: [{'entity': 'Sporting News', 'relationship': 'PUBLISHED_BY'}, {'entity': 'NFL', 'relationship': 'MENTIONS'}, {'entity': 'line shopping',
'relationship': 'MENTIONS'}, {'entity': 'odds and lines', 'relationship': 'MENTIONS'}, {'entity': 'How to bet on sports in the US', 'relationship': 'MENTIONS'},
{'entity': 'live odds feed', 'relationship': 'MENTIONS'}, {'entity': 'sportsbook apps', 'relationship': 'MENTIONS'}, {'entity': 'live odds feeds',
'relationship': 'MENTIONS'}, {'entity': 'totals bets', 'relationship': 'MENTIONS'}, {'entity': 'futures markets', 'relationship': 'MENTIONS'}, {'entity':
'liability', 'relationship': 'MENTIONS'}, {'entity': 'teams', 'relationship': 'MENTIONS'}, {'entity': 'Vegas line', 'relationship': 'MENTIONS'}, {'entity': 'Las
Vegas sportsbooks', 'relationship': 'MENTIONS'}, {'entity': 'online betting sites', 'relationship': 'MENTIONS'}, {'entity': 'parlay bet', 'relationship':
'MENTIONS'}, {'entity': 'roster move', 'relationship': 'MENTIONS'}, {'entity': 'Cincinnati Bengals', 'relationship': 'MENTIONS'}, {'entity': 'US',
'relationship': 'MENTIONS'}, {'entity': 'point spread', 'relationship': 'MENTIONS'}, {'entity': 'sportsbooks', 'relationship': 'MENTIONS'}, {'entity': 'sports
betting apps', 'relationship': 'MENTIONS'}, {'entity': 'Betting patterns', 'relationship': 'MENTIONS'}, {'entity': 'DraftKings Sportsbook', 'relationship':
'MENTIONS'}, {'entity': 'FanDuel Sportsbook', 'relationship': 'MENTIONS'}, {'entity': 'BetMGM Sportsbook', 'relationship': 'MENTIONS'}, {'entity': 'injury',
'relationship': 'MENTIONS'}, {'entity': 'Caesars Sportsbook', 'relationship': 'MENTIONS'}, {'entity': 'NBA', 'relationship': 'MENTIONS'}, {'entity': 'prop
bets', 'relationship': 'MENTIONS'}, {'entity': 'sports betting', 'relationship': 'MENTIONS'}, {'entity': 'Kansas City Chiefs', 'relationship': 'MENTIONS'},
{'entity': 'Las Vegas', 'relationship': 'MENTIONS'}, {'entity': 'point spreads', 'relationship': 'MENTIONS'}, {'entity': 'moneylines', 'relationship':
'MENTIONS'}, {'entity': 'Odds', 'relationship': 'MENTIONS'}]
passages: ['Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nLine shopping is the key that unlocks the
best odds and lines at top sports betting sites.\n\nThe process is as simple as checking the lines and odds for your prospective bet at multiple online sports
betting sites. The result is quite often better odds and improved lines.\n\nWhat is line shopping, and why is it worth your time? Find the answers below in our
complete guide to line shopping.\n\nHow do sportsbooks set odds and lines?\n\nThe best sportsbooks employ oddsmakers and oddsmaking services to set their lines
and odds.\n\nOther sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that consider
analytics, statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.\n\nBetting lines and odds are then subject to
change based on the betting public’s reaction and any additional information that comes to light ahead of the game. Sportsbooks will move the line or change the
odds if the public bets heavily on one side.', 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News,
2023-11-06T22:21:05+00:00)\nSportsbooks will move the line or change the odds if the public bets heavily on one side.\n\nEvery sportsbook’s goal is to even out
the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without worrying about which side wins.\n\nThe
betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t. That’s
when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.\n\nThe lines and odds
may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can find opportunities to
lay less, win more, and improve your chances.', 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nShop
around, and you can find opportunities to lay less, win more, and improve your chances.\n\nSee more: How to bet on sports in the US\n\nShop for the best lines
at online sportsbooks\n\nBelow is a live odds feed that shows you the differences between lines and odds for the same bets at many of the top sportsbooks in the
country. Look here for current opportunities.\n\nWhat’s the best way to shop lines?\n\nThe easiest way to line-shop is to find something like the live odds feed
above. It’s a one-stop shop. Shopping lines here will take just seconds.\n\nPerforming the task manually may take longer, but you can check with more
sportsbooks or the chosen few you feel consistently offer the best lines and odds. The best way to do this is to download and install more than one of the best
sports betting apps on your phone.\n\nThen, tap around among the sportsbooks to compare lines and odds for the bets you plan.\n\nBest sportsbooks for line
shopping\n\nThe biggest brands in sports betting offer competitive lines in dozens of states.', "Line Shopping in Sports Betting - How to Find the Best Odds
(Sporting News, 2023-11-06T22:21:05+00:00)\nShop around at the following top online sportsbooks to find the best prices on your bets.\n\nCaesars
Sportsbook\n\nCaesars' odds and lines are consistently among the best in the country, making it a must-see when shopping lines. The point spreads here can be a
half-point to two points better than the competition’s, and the moneyline odds may be more favorable.\n\nBetMGM Sportsbook\n\nThe lines and odds at BetMGM are
consistently competitive. Plus, sports betting promos and odds boosts abound, so you'll find a better price on many bets here.\n\nDraftKings
Sportsbook\n\nDraftKings has more open betting markets than most online sportsbooks, and the lines and odds are always competitive.\n\nFanDuel
Sportsbook\n\nFanDuel is committed to posting competitive odds and runs so many generous promos and odds boosts that it is always worth a look.\n\nHow to find
the best lines and odds for any bet\n\nShopping around lets you find the best lines and odds for any bet.", 'Line Shopping in Sports Betting - How to Find the
Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nThat means checking the posted lines and odds for the same bets at different sportsbooks.\n\nYou can do
this for moneylines, point spreads, totals bets, prop bets, and more. Finding the best line or better odds on any bet is like finding money.\n\nYou can shop
around using multiple sportsbook apps on your phone or live odds feeds like the one above. Pick a side, then use the drop-down menus and tabs to find the odds
and lines for that bet.\n\nFinally, bet when you see an opportunity at better odds or an improved line compared to the competition.\n\nLine shopping for
futures\n\nPeople tend to bet futures at their favorite online sportsbook. However, shopping for better odds on any future bet is worth your time. Most game
lines are similar at different sportsbooks, but the odds in futures markets can vary significantly.\n\nSportsbooks sometimes have higher liabilities for
specific teams and offer better odds on others to lay that off. Timing is everything.\n\nGet it right, and you can easily take advantage of changes in different
futures markets due to liability.', "Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nGetting more points
or laying fewer on a typical NFL point spread or NBA point spread might lead to winning a bet you’d otherwise lose. Plus, if you find better odds on any bet,
you can potentially drop fewer dollars when you lose and earn more when you win.\n\nShopping lines will help you consistently find the best prices on your bets.
It won't help you pick more winners, but it will improve your chances and lead to bigger wins and more affordable losses. That’s undoubtedly worth your
time.\n\nFor example, you might like the Kansas City Chiefs to beat the Cincinnati Bengals in a mid-season NFL game. Say you look around and see that three of
four licensed online sportsbooks in your state have the Chiefs as -3 favorites, but the fourth has them at just -2.5.\n\nLet’s say the Chiefs win by exactly
three, a distinct possibility since a single field goal decides most NFL games. Bets at the first three sportsbooks would push. However, you’d have a winner at
the fourth.", 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nHowever, you’d have a winner at the
fourth. If you didn’t shop around, you’d never have found that half-point victory and be stuck with a push instead of a profit.\n\nWhat is the Vegas
line?\n\nLas Vegas sportsbooks set the standard for lines and odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line.
Most online betting sites don’t stray too far from the Vegas line, using it as a foundation for setting all lines in every major sport.\n\nIf those lines do
shift, it’s usually because of increased betting on one side of a bet. Shop lines locally, and you can take advantage when sportsbooks stray from the Vegas
line.\n\nWhy do the lines move?\n\nIt’s worth reiterating that sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they
earn the small profit built into the odds without concern about who wins or loses.', 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting
News, 2023-11-06T22:21:05+00:00)\nSportsbooks want even betting so much that they’ll move a line or change the odds to draw more bets to the other side to get
it that way.\n\nThat’s when you’ll see a difference between the lines and odds. Every sportsbook jumps on board when a line moves or odds change for a
legitimate reason, like an injury or roster move. It’ll be difficult for you to find value in that case.\n\nBut when a line moves or odds change because of
betting patterns, only the sportsbook that saw the betting makes the change. That’s your opportunity right there.\n\nLine shopping is a tool to help you spot
changes due to betting patterns. Then, you can simply take advantage of movement in a line or changes in the odds.\n\nChoose a side, shop around at online
sportsbooks, compare the lines and odds, and then pounce when you find one that has something different and more favorable to your viewpoint.\n\nSee more: What
is a parlay bet?']
</item 9>

<item 10>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
What makes NBA betting special or unique?

Basketball odds are known to fluctuate more than those of other sports. The factors that play into those lines shifting can range from having only a few impact
players per team to the sheer frequency of games.

By following a single team, you can track its momentum swings, which players tend to get on hot or cold streaks, or how well they play at home vs. on the road.
You can use that information to your advantage for both pre-game and live wagers.

Similar categories of moneylines, spreads, totals, and NBA prop bets should appear across both formats.

The NBA is also special because of the number of bets available for every game, ranging from pre-game to live betting and parlays.

When you add up the standard bets, game props, player props, in-game betting, and more, there’s a staggering volume of bets for every NBA game at online
sportsbooks.
</item 10>
=== END ITEM id=7bc805783aa8bd37 ===

=== ITEM id=f611e4eeddb27fc1 ===
Gold sentences and facts:
Sentence 0: They’d all have significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.
  Fact 0.0: Jerry Dipoto is president of baseball operations.
  Fact 0.1: Those players would all have significant value if Jerry Dipoto decided to move one for a bat.
Sentence 1: Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.
  Fact 1.0: Aaron Judge came away with a new contract.
  Fact 1.1: Justin Verlander came away with a new contract.
  Fact 1.2: Trea Turner came away with a new contract.
  Fact 1.3: Xander Bogaerts came away with a new contract.

Retrieved items:
<item 1>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
His 3.39 mark across 193 2/3 innings this past season wasn’t quite so dominant but is still high-end production.

Burnes struck out just under 26% of opposing hitters. While his lowest mark since his rookie season, that’s still an above-average figure.

Adames is coming off a less impressive year. The 28-year-old hit .217/.310/.407 through 638 trips to the plate. That’s slightly below-average offense, although
he still connected on 24 home runs.

Adames is an excellent defensive shortstop with plus power in an offseason where there are essentially no free agent shortstops of note. Even with a mediocre
on-base percentage, he’d generate plenty of interest.

4. Shane Bieber, SP, Guardians

Develop high-end starting pitching, trade a top starter as they become more expensive, replace him with talented younger arms. The Guardians have trodden this
path with Corey Kluber, Mike Clevinger and Carlos Carrasco.
</item 1>

<item 2>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
3

Source: Susan Slusser, San Francisco Chronicle

The Giants have missed out on big names in the past like Aaron Judge and Carlos Correa, but they are looking to turn that around in 2023 with heavy involvement
and Ohtani, among others. Per San Francisco Chronicle's Susan Slusser, teams believe the Giants "might have an edge" in the pursuit of Japanese ace Yoshinobu
Yamamoto.

Dylan Cease trade market heating up

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

One of the biggest names on the market early is White Sox ace Dylan Cease in trade talks. MLB Network's Jon Morosi reported three teams, the Dodgers, Braves and
Orioles, are among several involved discussing a deal for Cease as the winter meetings begin.

Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec.
</item 2>

<item 3>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
MLB's winter meetings are always a prime time for moves to happen as many of the top executives, players and agents all come together at the same spot.

Last year's winter meetings wound up being explosive. Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.

This year's winter meetings could wind up being similarly busy. There is a growing sense Shohei Ohtani could sign during the winter meetings, and a move of that
magnitude could kick off the next wave of moves. There's also a number of top players on the trade market, including star Padres outfielder Juan Soto.

Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.

WINTER MEETINGS: Shohei Ohtani rumors, Juan Soto trade destination come to forefront

Sporting News will be tracking all the latest news and rumors from the 2023 Winter Meetings.
</item 3>

<item 4>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Verdugo had been the Red Sox's starting outfielder for the last four seasons, coming over as the centerpiece in the trade that sent Mookie Betts to the Dodgers.
Verdugo hit .281 with 43 home runs, 206 RBI and 16 stolen bases over 2,071 plate appearances with Boston. He also led the American League in outfield assists in
2020 and 2023.

In return, Boston will get Weissert and two pitching prospects headlined by Fitts — the Yankees' No. 4-ranked pitching prospect and 12-ranked prospect overall.

Relievers Yates, Devenski get one-year deals for 2024

Date: Tuesday, Dec. 5

Source: Ken Rosenthal, The Athletic and Jeff Passan, ESPN

Veteran relievers Kirby Yates and Chris Devenski know where they'll be pitching in 2024, according to reports Tuesday.
</item 4>

<item 5>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Davis, Mike Yastrzemski

Guardians: Josh Naylor, Cal Quantrill

Mariners: Ty France

Marlins: Josh Bell

Mets: Jeff McNeil

Nationals: Hunter Harvey, Lane Thomas

Orioles: Anthony Santander

Padres: Scott Barlow

Phillies: Nick Castellanos

Rays: Manuel Margot, Harold Ramírez

Red Sox: Kenley Jansen, Chris Martin, Nick Pivetta

Rockies: Brendan Rodgers

Royals: Salvador Perez

Tigers: Spencer Turnbull

Twins: Kyle Farmer
</item 5>

<item 6>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Twins could trade Max Kepler, Jorge Polance

Date: Sunday, Dec. 17

Source: Jon Morosi, MLB Network

The Twins are coming off an AL Central title, but could be looking to deal away two of their key players form the team. Max Kepler and Jorge Polanco could both
reportedly be on the trade market, per MLB Network's Jon Morosi, though their market could hinge on the status of other infield free agents.

Kepler appeared in 130 games for the Twins, and posted an impressive .260/.332/.484 slash line with 24 homers for Minnesota. Polanco was moved to an infield
utility role with Edouard Julien and Royce Lewis emerging into regulars, but was still solid in 80 games, logging a .255/.335/.454 slash with 14 homers.

Yamamoto requests meeting with Yankees

Date: Sunday, Dec.
</item 6>

<item 7>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
While the offseason has gotten rolling, trade season hasn’t quite arrived. The Tigers acquired Mark Canha from Milwaukee in the biggest swap of the first couple
weeks. Otherwise, the GM Meetings simply served to set the stage for future transactions.

That’ll surely involve some notable trades. Unlike the summer deadline, projecting which players will change hands is difficult at this stage of the offseason.

Aside from the A’s, there aren’t any teams clearly in rebuild mode. Clubs like the Rockies, Royals, White Sox, Nationals and Angels have uphill paths to
contention, yet none has indicated they’re prepared to punt on the 2024 season entirely.

The Padres, Twins, Rays, Brewers and Guardians are all aiming to contend but have questions about their payroll. That’s standard operating procedure for Tampa
Bay, Milwaukee and Cleveland. San Diego and, to a lesser extent, Minnesota have shown a willingness to spend in recent years but are scaling back — in both
cases, likely tied to their uncertain local television rights contracts.
</item 7>

<item 8>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

The Mariners and Rays have been frequent trade partners in recent years, and there's a chance they could be talking shop again at the winter meetings. MLB
Network's Jon Morosi reported the teams are discussing Rays utility infielder Isaac Paredes, and potentially star left fielder Randy Arozarena, in talks to
start the winter meetings.

Jays unlikely to land both Ohtani, Soto

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

The Blue Jays have been among the most active names in both the Ohtani and Soto markets. And of course, those talks have had fans dreaming of a lineup that
features, Ohtani, Soto, Vladimir Guerrero Jr., Bo Bichette and George Springer.
</item 8>

<item 9>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
16. Mariners SP

We initially had Bryan Woo in this spot. It could just as easily have been Bryce Miller or Emerson Hancock, so we’ll cheat and just mention the Seattle rotation
in general.

They’re almost certainly not going to trade George Kirby. It’d take a lot to pry away Logan Gilbert. Moving any of Woo, Miller or Hancock could be more
appealing for a front office that is looking for ways to add long-term offensive upside.

None of that trio has even reached one year of major league service. Hancock only has three big league starts under his belt. Miller and Woo spent more time in
the rotation this year, starting 25 and 18 games respectively. They both turned in mid-rotation results.

Other clubs probably have differing opinions on which right-hander they prefer. (The Mariners likely have their own slight preferences.) They’d all have
significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.

17.
</item 9>

<item 10>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
It also makes him less likely to be traded this offseason.

First-year GM Chris Getz has suggested no one on the roster is truly untouchable, an understandable approach for a team that just lost 101 games. Neither owner
Jerry Reinsdorf nor Getz has called this a rebuild, though.

Any slim hopes the White Sox may have of competing in 2024 would be dashed by trading Cease, the only above-average starting pitcher on the roster.

7. Alex Verdugo, RF, Red Sox

Verdugo’s name has been floated in trade rumors for a few seasons. The Red Sox have held onto him thus far. They have a new baseball operations leader in Craig
Breslow, who has acknowledged the team has gotten early calls on the left-handed hitting outfielder.

Jarren Duran and Masataka Yoshida also hit from the left side. Dealing Verdugo could free some at-bats for a right-handed addition while perhaps bringing in
immediate help for the pitching staff or at second base.
</item 10>
=== END ITEM id=f611e4eeddb27fc1 ===

=== ITEM id=fc80a0d5cb1b2dea ===
Gold sentences and facts:
Sentence 0: Microsoft also has a stripped-down Elite Series 2 “Core” version for $129.99, which is the same controller in white-and-black, red-and-black, or blue-and-black finishes without the rear paddles, accessory charging case, or extra stick toppers and D-pad.
  Fact 0.0: Microsoft also has a stripped-down Elite Series 2 'Core' version priced at $129.99.
  Fact 0.1: The Elite Series 2 Core is the same controller in white-and-black, red-and-black, or blue-and-black finishes.
  Fact 0.2: The Elite Series 2 Core version lacks the rear paddles.
  Fact 0.3: The Elite Series 2 Core version lacks the accessory charging case.
  Fact 0.4: The Elite Series 2 Core version lacks extra stick toppers and D-pad.
Sentence 1: While these models are still quite new, versions that actual humans buy have begun receiving discounts.
  Fact 1.0: These models are still quite new.
  Fact 1.1: Versions that actual humans buy have begun receiving discounts.

Retrieved items:
<item 1>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
But it differs by having a larger, 15.3-inch notched screen with 2880 x 1864 resolution, six speakers instead of four, a 10-core GPU that costs extra on the
13-inch, and a higher starting price of $1,299. Bumping up to a larger, faster 512GB of storage will run you $1,499.

Surprisingly, the 15-inch M2 Air started receiving some discounts even before it shipped, which let early adopters save between $50 and $100. As of right now,
you can get it $300 off by ordering it at Amazon or Best Buy, where it’s currently selling for $999. The expanded 512GB model is also $300 off at Best Buy,
selling for $1,199.

The best MacBook Pro deals

The MacBook Pro line is once again in a transitional phase.
</item 1>

<item 2>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
You can, of course, spec them up the wazoo if you’re willing to pay more, as exhibited by the review unit Apple sent us of the 16-inch model, which cost an
eye-watering $7,199.

While these models are still quite new, versions that actual humans buy have begun receiving discounts. The 14-inch MacBook Pro in its base configuration with
the M3 Pro (11-core CPU / 14-core GPU), 18GB of RAM, and 512GB SSD is currently on sale for $1,749 ($250 off) at B&H Photo. A similar version with 1TB of
storage and a 12-core / 18-core M3 Pro is down to $2,149 (also $250 off) at B&H Photo. As for the base 16-inch MacBook Pro with the M3 Pro (12-core CPU /
18-core GPU), 18GB of RAM, and a 512GB SSD, it’s selling for $2,249 ($250 off) at Amazon and B&H Photo.
</item 2>

<item 3>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
But if you want to buy new and you’re looking to save whatever you can, here are the best MacBook Air, MacBook Pro, and Mac Mini deals available right now.

A note about discounted Intel-based Macs At this point, we don’t recommend buying any older Intel-based Macs. Retailers may try to tempt you with very low
prices on leftover new or refurbished stock, but there’s a reason nobody wants them anymore. They’re completely outclassed by all the newer models, unless you
have some super-niche Windows Boot Camp needs — and even then, consider what Windows apps you really need and if they work with Parallels.

The best MacBook Air deals

M1 MacBook Air

The MacBook Air is Apple’s entry-level laptop. It’s best suited for typical productivity work, with a comfortable keyboard, an excellent trackpad, and all-day
battery life.
</item 3>

<item 4>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
That said, if you plan on going with a higher-spec configuration than that, you’ll likely want to spring for the M3 Pro model instead (see below).

The 14- and 16-inch MacBook Pros for late 2023 are another round of spec-bump models, much like their previous 2022 incarnations. Now starting with the M3 Pro
chip (or the speedier M3 Max) and 18GB of base RAM instead of 16GB, the new models remain targeted at creatives doing content work like video editing, photo
processing, and other graphical work. Like their predecessors and the M1 Pro generation before that, they offer MagSafe charging, three USB-C / Thunderbolt 4
ports, HDMI-out, and a full-size SD card slot, with prices starting at $1,999 for the 14-inch and $2,499 for the 16-inch.
</item 4>

<item 5>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
M2 Pro MacBook Pro

As for the 14- and 16-inch MacBook Pros from early 2023 with M2 Pro / Max processors, they’re still more than capable — they may even be a better buy than the
new models if you can find them at a steep discount.

For example, the base 14-inch MacBook Pro with an M2 Pro chip, a 512GB SSD, and 16GB of RAM is selling for $1,629 ($370 off) at Amazon, which is a new low price
that may not last very long. However, the 14-inch model with 1TB of storage, a 12-core M2 Pro CPU, and 19 GPU cores is down to $1,999 (a steep $500 off) at B&H
Photo. Now we’re talking!

Sadly, the situation isn’t quite as good for the 16-inch MacBook Pro with the M2 Pro.
</item 5>

<item 6>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
However, using that many buttons on the back of the controller definitely takes some adjustment. You have to train new muscle memory, and since it requires
dexterity with more fingers, they’re not for everyone.

These premium features come at a steep cost, too. The Elite Series 2 is normally priced at $179.99, and even though it occasionally drops to $139.99, it’s still
more than twice as expensive as a regular Xbox controller. You have to really want its extra features and user customization, and if you’re not using its rear
paddles or the software adjustments for stick sensitivity, you’re not getting your money’s worth. (And if you want to customize the colors, you’re looking at
$209.99 for the Design Lab “Elite Package.”)

Microsoft also has a stripped-down Elite Series 2 “Core” version for $129.99, which is the same controller in white-and-black, red-and-black, or blue-and-black
finishes without the rear paddles, accessory charging case, or extra stick toppers and D-pad.
</item 6>

<item 7>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The best MacBook Pro deals

The MacBook Pro line is once again in a transitional phase. Apple announced Macs using a new M3 processor at its “Scary Fast” event the night before Halloween,
including a new 14-inch MacBook Pro that replaces the 13-inch M2 model. That means Touch Bar Macs are finally dead, but some of those outgoing M2 models are
still available for now. You’ll find any deals on them further down, but first, let’s focus on the new M3 generation.

M3 and M3 Pro MacBook Pro

The new entry point into the MacBook Pro world is a MagSafe-equipped MacBook Pro that uses the existing 14-inch design and slightly pares it down. The 14-inch
M3 MacBook Pro has the same 3024 x 1964 resolution display and 120Hz refresh rate as its pricier siblings, and in addition to a MagSafe charging port, it has an
SD card slot and HDMI port.
</item 7>

<item 8>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
Apple sells MacBooks equipped with its own M-series chips in a wide range of sizes and price points. The offerings start with the 13-inch MacBook Air from 2020
at $999 and go all the way up to the latest 16-inch MacBook Pro starting at $2,499. But finding a deal on a current Mac with an M1, M2, or even the new M3 chip
— as well as the higher-end M3 Pro and M3 Max — is actually not that difficult.

While Macs may not experience perpetual discounts, it’s not uncommon to see various current models discounted by as much as $400. Alternatively, purchasing
refurbished options directly from Apple is another way to save money without as much waiting for the changing winds of deals to blow your way. Apple’s
refurbished store provides a one-year warranty on all products and generally offers discounts of up to 15 to 20 percent off the price of a new unit.
</item 8>

<item 9>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The best (and only) deal available right now on the creator-friendly laptop is at Amazon, where you can grab the base M2 Pro model for $2,299.99 ($200 off) with
512GB storage. That’s not great, but hopefully we see better deals before the laptop is no longer available new.

M1 Pro and M1 Max MacBook Pro

For now, the older 2021 MacBook Pros can occasionally be had for some steep discounts that actually make a two-generation-old laptop still worth considering.
You’re unlikely to find any base models still kicking around, but if you want a souped-up configuration on the cheap, you may be in luck.

The 16-inch MacBook Pro from 2021 with 16GB of RAM and an expanded 1TB of storage is available at B&H Photo for $1,799 ($900 off).
</item 9>

<item 10>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
However, it starts with just 8GB of RAM and lacks the third USB-C / Thunderbolt port found on the M3 Pro and M3 Max models (as well as Thunderbolt 4 speeds).

The M3 MacBook Pro may be a bit of an odd middle child in some ways, but it’s still a very good laptop — especially if you can get it on a deal that puts more
price distance between it and the M3 Pro version. We’ve already seen the base model with 512GB of storage on sale for $150 off during Black Friday and Cyber
Monday, and right now, it’s actually available for an even better $1,399 ($200 off) at Amazon, Best Buy, and B&H Photo. The 1TB configuration is also selling
for its all-time low of $1,599 ($200 off) at Best Buy and B&H Photo, while the configuration with base storage and 16GB of RAM is selling for $1,599 ($200 off)
at B&H Photo.
</item 10>
=== END ITEM id=fc80a0d5cb1b2dea ===
