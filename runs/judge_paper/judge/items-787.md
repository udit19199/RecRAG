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

=== ITEM id=75c14bd2ff5c108d ===
Gold sentences and facts:
Sentence 0: You can find value in player and team props by gathering information from multiple sources and tracking how players and their teammates perform.
  Fact 0.0: Value can be found in player props.
  Fact 0.1: Value can be found in team props.
  Fact 0.2: Value can be found by gathering information from multiple sources.
  Fact 0.3: Value can be found by tracking how players and their teammates perform.
Sentence 1: You can also bet on team props, such as whether a team will lead at the end of a certain quarter or an over/under bet on the total number of points the team will score in the game.
  Fact 1.0: Bets can be placed on team props.
  Fact 1.1: A team props bet can be whether a team will lead at the end of a certain quarter.
  Fact 1.2: A team props bet can be an over/under bet on the total number of points the team will score in the game.

Retrieved items:
<item 1>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Accept the terms and conditions: Before finalizing your account information, you must accept the sportsbook’s terms and conditions. We recommend you read
through these to understand the rules your sportsbook will enforce regarding bets, especially for the sports you intend to wager on. The terms and conditions
are also available on the sportsbook website for you to read at any point.

Get your bonus: At this point, you can accept your welcome bonus. Be sure to read through the bonus requirements before accepting anything. If a bonus code is
needed, which you can find on our website, this is where you will input it during the registration process.

Make a deposit: Now that you’ve signed up for an account, you can start funding your bankroll by making a deposit. Most legal sportsbooks have a variety of ways
you can add funds to your account, such as through your bank or an e-wallet.

Start betting: Go to the basketball section of your sportsbook and select the NBA.
</item 1>

<item 2>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Positive numbers, like +145, show how much you stand to win for every $100 you wager. If you were to bet $50 on Giannis Antetokounmpo to have over 7.5 rebounds
and the odds were +130 at the time you finalized your bet slip, you would win $65 in profit if your bet wins.

Positive numbers also indicate which side of the bet is the underdog in the eyes of the oddsmakers.

Remember that you can bet as little or as much as you want, and your payout will be calculated based on your wager amount and the odds at the time you make your
bet.

Types of NBA props

You can make two main prop bets on the National Basketball Association:

Team props

Player props

Team props are typically focused on statistical accomplishments by a particular team and rarely on the game’s result.
</item 2>

<item 3>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Other prop bets can be focused on the team or even both teams.

See more: Super Bowl prop bets

MLB prop bets

MLB betting sites are no exception when it comes to offering prop bets. Like the other sports, MLB has prop bets focused primarily on individual players and
statistical achievements like home runs, RBI, and strikeouts.

You might bet on an over/under for strikeouts for a starting pitcher in a game or even the number of bases a hitter will reach.

NHL prop bets

NHL betting sites have all of the props you’d expect from one of the major North American sports leagues, such as player assists or a certain player scoring a
goal in a contest.

You will also find bets for shots on goal, team goals in a game, and which team will score first.

MLS prop bets

The demand for Major League Soccer betting has increased as the sport's popularity continues growing nationwide. Prop bets are just one of the many bet types
growing at sportsbooks to keep up with the demand.
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
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.
</item 5>

<item 6>
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
</item 6>

<item 7>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Let’s say you’re keeping an eye on the New York Jets vs. the Buffalo Bills, and you see the total shift due to an incoming winter storm. While many bettors will
scramble to get their bet in on the ‘under’ before the lines move too much, it’s a good idea to scope out the prop bets for individual players. The main lines
are shifting, but most of the best sports betting sites don’t have the personnel to keep tabs on every possible player, so you could find value in an ‘under’ on
a quarterback’s passing yards because the lines haven’t caught up yet. There’s greater variety: The three main bet types—moneylines, point spreads, and
totals—are constant in every major sport, but they don’t offer a lot in the way of variety. Prop bets do. They can add interest to a game that may seem
one-sided in the major markets. Larger sporting events, especially the Super Bowl, offer a massive variety of bets, ranging from the typical to the exotic and
everything in between.
</item 7>

<item 8>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Remember that some of the best players in the NBA were once on the bench for most games until small increases in their playing time gave them a chance to shine.

Shop for the best lines: This tip applies to every bet you make. Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the
smallest differences, like one sportsbook having odds at +125 and another at +135, can make a difference in your profit. By shopping for the best lines, you can
make smarter bets and find the best value.

Related Pages:
</item 8>

<item 9>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
But if that player is involved in the game even for a single play, then the bet stands.

We recommend reading through the T&Cs before placing any wagers so you’re not caught by surprise if an unexpected event should happen.

Related Pages:
</item 9>

<item 10>
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
</item 10>

<item 11>
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
</item 11>

<item 12>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Pros

The lines don’t usually move as fast as other bet types: Prop bet odds are developed based on game lines and totals at sportsbooks, but prop bet lines tend to
adjust more slowly. Let’s say you’re keeping an eye on the New York Jets vs. the Buffalo Bills, and you see the total shift due to an incoming winter storm.
While many bettors will scramble to get their bet in on the ‘under’ before the lines move too much, it’s a good idea to scope out the prop bets for individual
players. The main lines are shifting, but most of the best sports betting sites don’t have the personnel to keep tabs on every possible player, so you could
find value in an ‘under’ on a quarterback’s passing yards because the lines haven’t caught up yet.

Prop bet odds are developed based on game lines and totals at sportsbooks, but prop bet lines tend to adjust more slowly. Let’s say you’re keeping an eye on the
New York Jets vs.
</item 12>

<item 13>
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
</item 13>

<item 14>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Start betting: Go to the basketball section of your sportsbook and select the NBA. From there, you can choose individual games and make bets like money lines,
point spreads, totals, props, futures, and more.

How to read NBA props odds

At first glance, all the positive and negative numbers at US sportsbooks can be daunting. The good news is they are easy to read once you understand what you
see.

Negative numbers, such as -110, represent how much money you need to wager to win $100. In many cases, such as a moneyline bet, the negative number also
indicates which side of the bet is the favorite to win.

For example, if you were to bet on LeBron James scoring over 15.5 points in a contest at -125 odds, you’d need to wager $125 to win $100. If you did and your
bet was correct, then you’d take home $225, which is your original $125 stake returned to you plus your $100 in profit.
</item 14>

<item 15>
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
</item 15>

<item 16>
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
</item 16>

<item 17>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
The three main bet types—moneylines, point spreads, and totals—are constant in every major sport, but they don’t offer a lot in the way of variety. Prop bets
do. They can add interest to a game that may seem one-sided in the major markets. Larger sporting events, especially the Super Bowl, offer a massive variety of
bets, ranging from the typical to the exotic and everything in between. They are usually very easy to understand: Prop bets tend to ask a question of the
bettor. Will a specific player hit a home run? Will Team X have over/under 3.5 field goals in a game? Will Team Y give up a powerplay goal? Whatever the
question, prop bets are fairly easy to understand for the newcomer.

Cons

Props are difficult to research: Prop bets are so specific you’d think it would be easier to research them than other bet types, but the reality is that sports
are extremely volatile. You can never truly predict how an athlete or a team will perform.
</item 17>

<item 18>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
The NBA is one of the most popular sports to bet on in the US. NBA prop bets are an easy way to jump into basketball betting.

Proposition bets, or props, are easy to understand, numerous, and cover both player props and team props to give you plenty of variety.

Below, we look at betting NBA props at the best NBA betting sites and share some key information to make informed bets.

What are NBA props?

A prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result. In the NBA, prop bets deal with both players and teams.
</item 18>

<item 19>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
You can never truly predict how an athlete or a team will perform. This makes it very hard to nail down smart prop bets. For example, you could bet on a certain
player to score ‘over’ a point threshold, but you lose because that player had an off-night, even though they might have easily won that bet in other recent
games.

Prop bets are so specific you’d think it would be easier to research them than other bet types, but the reality is that sports are extremely volatile. You can
never truly predict how an athlete or a team will perform. This makes it very hard to nail down smart prop bets. For example, you could bet on a certain player
to score ‘over’ a point threshold, but you lose because that player had an off-night, even though they might have easily won that bet in other recent games.
Prop bets can tie up your bankroll: We wouldn’t recommend using prop bets as your main bet type when building your strategy.
</item 19>

<item 20>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
As expected, these prop bets often focus on individual players and their accomplishments, such as scoring a goal or collecting an assist in a game. Soccer
betting sites also have props such as how many cards the two teams playing will combine to draw or a team’s total number of corner kicks.

How to read prop bet odds

Prop bets at US sportsbooks use the same odds as other major bet types like moneylines, point spreads, and totals. These odds, known as American odds, use
negative and positive numbers to show you how much money you stand to win.

Positive odds show how much you stand to win for every $100 you wager. Let's say you wager on a quarterback to have over 2.5 touchdowns in a game at +145 odds,
and he throws three in the game. If you wagered $100, you would be paid $245, your initial wager of $100 returned to you, plus $145 in winnings.

Negative odds tell you how much you need to wager to win $100.
</item 20>
=== END ITEM id=75c14bd2ff5c108d ===

=== ITEM id=f40d129c5d769b5c ===
Gold sentences and facts:
Sentence 0: Instead, Marshall will start freshman Cole Pennington, who is the son of former Marshall star QB Chad Pennington.
  Fact 0.0: Instead, Marshall will start freshman Cole Pennington.
  Fact 0.1: Cole Pennington is a freshman.
  Fact 0.2: Cole Pennington is the son of former Marshall star QB Chad Pennington.
  Fact 0.3: Chad Pennington is a former Marshall star QB.
Sentence 1: Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.
  Fact 1.0: Geno Smith's night appears to be done.
  Fact 1.1: Drew Lock is back in for the Seahawks.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Three of the seven teams passed, resulting in one first down, one sack and Hurts’ interception. Of these three teams, only the Eagles lost the game.

3. The 49ers lost Sunday when their rookie kicker missed from 41 yards. Is San Francisco wise trusting a championship roster to a rookie’s foot come playoff
time?

The 49ers raised eyebrows among some when they used their second-highest pick in the 2023 draft — a third-round choice, 99th overall — for a kicker. The
selection of Michigan’s Jake Moody gave the 49ers a cost-effective successor to veteran Robbie Gould. General manager John Lynch called Moody a potentially
“foundational” player whose stock in the draft was so high, other teams were trying to trade ahead of San Francisco for a shot at him.

Who do you want lining up the potential winning kick in a Super Bowl?
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• San Francisco 49ers: What do they have in Purdy?

What I think: Let’s start with what the 49ers do not have in Purdy. They do not have an AFC North-certified, weatherproof Ben Roethlisberger clone able to
overpower the elements, break away from defensive linemen and thrive in the conditions Purdy faced against Cleveland. What they do have is a good, young
quarterback well-suited to run Kyle Shanahan’s offense at a high level by making quick decisions when given time to find open receivers.

Exec comment: “The 49ers knew they needed a good quarterback to win the Super Bowl, and they knew they could help Purdy be that. Give him a great receiving
tight end, two good receivers and a great halfback who fits the system, and they’ll get someone open on every pass play. But then Deebo Samuel gets hurt,
Christian McCaffrey gets hurt, and now we don’t have as many guys open.
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Payton is the only other coach with more than one such game over the past three-plus seasons.

The irony is rich in Denver, where Payton’s predecessor, Nathaniel Hackett, was mocked for his game management so thoroughly, the team made him hire veteran
assistant Jerry Rosburg to help him after only three games (Denver was 2-1 at that point). The Broncos under Hackett had trouble getting out of the huddle on
time.

Hackett took the hit for those issues, but when the problem recurred under Payton, the blame shifted to Wilson, the common denominator.

Advertisement

All of this is pointing toward a future in Denver that includes Payton but not Wilson. Payton’s game plans over the past two weeks were conservative enough to
suggest the coach was publicly demonstrating he does not think he can run his offense as currently configured, despite some success on the ground. These game
plans were conservative enough to signal a potential in-season quarterback change.
</item 4>

<item 5>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Which QBs are on bye in Week 14?

Coinciding with the end of the fantasy regular season, Week 14 is the last week of byes, as the Cardinals and Commanders are the last two teams to take their
mandatory PTO. That means two starting-caliber QBs (Kyler Murray and Sam Howell) are on the shelf, slightly impacting the QB pool.

WEEK 14 PPR RANKINGS:

RBs | WRs | TEs | D/STs | Kickers

The bigger losses, though, come on the defensive side of the ball, as fantasy owners won't be able to start QBs going up against two subpar pass defenses in
Washington and Arizona. Starting any QB facing the Commanders' swiss-cheese defense has yielded strong returns. Arizona has been just as inefficient, entering
Week 13 ranking 30th in dropback EPA.
</item 5>

<item 6>
Michigan State hires Jonathan Smith: Oregon State coach leaves alma mater to take over Spartans job (CBSSports.com, 2023-11-25T23:12:10+00:00)
"All of Beaver Nation should thank him for the lasting legacy he has left to his alma mater and wish him and his family nothing but the best moving forward."

Smith held multiple assistant jobs from 2002-17 before returning to his alma mater as coach in 2018. Smith spent six seasons as an assistant under former
college football coach Chris Petersen, first as Boise State's quarterbacks coach from 2012-13 and then as Washington's quarterbacks coach and offensive
coordinator from 2014-17. His time in Seattle included a Pac-12 title and College Football Playoff berth for Washington during the 2016 season.

Smith's exit from Oregon State comes at a point where the Beavers face much uncertainty as the Pac-12 prepares to lose 10 of its 12 members ahead of the 2024
season. Oregon State and Washington State are the only schools in the conference that have not been picked up by other power leagues.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 7>

<item 8>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Not far behind sits Tua Tagovailoa (vs. Titans), who shredded the Commanders' defense to the tune of 280 yards and two TDs in Sunday's blowout. Another juicy
matchup awaits against a lackluster Tennessee coverage unit. Dak Prescott (vs. Eagles) remains a top-three play in Sunday's rematch versus the Eagles, a team he
torched for 374 yards and three TDs in a narrow 28-23 loss in Week 9.

Jalen Hurts (@ Cowboys) and the Eagles look to get back on track in a pivotal NFC East clash in Dallas, facing a Cowboys defense that let Geno Smith post
top-end QB1 numbers against them last week. Patrick Mahomes (vs. Bills) is in line to put up strong numbers in a potential shootout against Buffalo, and
although his receiving corps has its issues, the reigning MVP is still a top-five play this week.

Lamar Jackson (vs.
</item 8>

<item 9>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
UTSA is 5-6-1 ATS and 4-7-1 to the OVER.

UTSA Roadrunners opting out of Scooter's Coffee Frisco Bowl

The Roadrunners will be without their best defensive player, Trey Moore, as he looks to jump to a Power 5 school. Moore led the conference in sacks, so he's a
star-level absence for the unit.

Here are USTA players who won't play in the Frisco Bowl.

Trey Moore, EDGE (transfer portal)

Syrus Dumas, DB (transfer portal)

Xavier Spencer, CB (transfer portal)

Kyle Eaves, WR (transfer portal)

Marshall Thundering Herd betting news

The toughest blow for Marshall in this game is being without second-year starting QB Cam Fancher (transfer portal). Instead, Marshall will start freshman Cole
Pennington, who is the son of former Marshall star QB Chad Pennington.
</item 9>

<item 10>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Assuming Winston gets the starting nod in place of the banged-up Derek Carr (head, shoulder, back), his gunslinger play style could lead to a respectable
fantasy output against the reeling Panthers.

Mayfield wasn't asked to do much with his arm in Week 13, but a potential negative game script in Atlanta could force him to air the ball out at a higher clip,
boosting his fantasy prospects. Although the Browns didn't get the win in Flacco's first start with his new team, the veteran did everything he could to keep
Cleveland within striking distance, racking up 254 yards and two TDs. Another 250-plus yard, multi-TD effort could again be in store if he continues to start in
place of Dorian Thompson-Robinson (concussion).

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the biggest potential QB busts in Week 14?
</item 10>

<item 11>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 11>

<item 12>
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
</item 12>

<item 13>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 13>

<item 14>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Stroud all have more potential to help you win.

LAST WEEK: After an interception, Goff marched downfield and got the Lions a touchdown, then gave way to David Montgomery to pound the Packers into submission.
He had his lowest number of pass attempts (28) and yards (210) this year, and threw just one touchdown for the third time in four games.

PANTHERS: At 27.9% are dead last in pass rush pressure rate through four weeks. The only game this season when they had some pressure on a quarterback was
against the Seahawks, who have a beat-up offensive line. The Lions offensive line is near full strength and have allowed Goff to be sacked five times through
four games.

QUARTERBACKS: Have failed to top 20 Fantasy points against the Panthers this season. That includes Kirk Cousins last week, who only threw 19 times in a win (and
had two touchdowns but just 13 Fantasy points).
</item 14>

<item 15>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 15>

<item 16>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Broncos

12 Russell Wilson, Broncos @ Chargers

13 Justin Fields, Bears vs. Lions

14 Gardner Minshew, Colts @ Bengals

15 Matthew Stafford, Rams @ Ravens

16 Trevor Lawrence, Jaguars @ Browns

17 Geno Smith, Seahawks @ 49ers

18 Derek Carr, Saints vs. Panthers

19 Joe Flacco, Browns @ Jaguars

20 Baker Mayfield, Buccaneers @ Falcons

21 Josh Dobbs, Vikings @ Raiders

22 Jake Browning, Bengals vs. Colts

23 Bryce Young, Panthers @ Saints

24 Aidan O'Connell, Raiders vs. Vikings

25 Desmond Ridder, Falcons vs. Buccaneers

26 Will Levis, Titans @ Dolphins

27 Bailey Zappe, Patriots @ Steelers

28 Mitch Trubisky, Steelers @ Patriots

29 Tommy DeVito, Giants vs. Packers

30 Zach Wilson, Jets vs. Texans
</item 16>

<item 17>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 17>

<item 18>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 18>

<item 19>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The 2017 Eagles won a Super Bowl with rookie Jake Elliott performing well. The 2012 Super Bowl champion Ravens had rookie Justin Tucker, while the 2006 Patriots
trusted rookie Stephen Gostkowski.

4. Do they make wristbands for head coaches? That is not the only question facing the Broncos.

Denver Broncos coach Sean Payton has made clear his frustration with quarterback Russell Wilson in myriad ways, including when he suggested Wilson could at some
point wear a wristband if it would help speed up communication in the huddle.

We joked on the Football GM Podcast that the head coach himself could have used a wristband to assist with game management after the Broncos called timeout
before punting with 22 seconds left in the first half of a 19-8 defeat at Kansas City in Week 6. Payton said he thought Denver was heading into third down, not
fourth, when he called timeout, helping the Chiefs’ drive to a field goal.

It wasn’t an isolated miscue.
</item 19>

<item 20>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.
</item 20>

<item 21>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Helmet technology

CBSSports.com reported on Oct. 20 that the NCAA rules committee is giving teams the option of coach-to-player helmet communication to relay plays from the
sideline during bowl season this year. Tablets will inevitably follow, and here is the sliver of good that will come from this scandal.

It's about damn time. Death to sandwich-board playcalling, please. Bring on those playsheets where coaches simply have to cover their mouth. This is one
instance where emulating the NFL game is overdue.

Sherrone Moore

If Moore is going to be a head coach, then Saturday’s victory against the Nittany Lions was a fitting audition. The Wolverines rushed for 227 yards – including
32 straight run plays – in grinding out a top-10 victory at one of the toughest environments in the FBS at Beaver Stadium.

Moore was emotional - complete with NSFW language – in the aftermath.

Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.
</item 21>

<item 22>
Hottest seat on each NFC team: Buns burning for these four head coaches (Yardbarker, 2023-11-30T22:29:33+00:00)
SAN FRANCISCO 49ERS | Head coach Kyle Shanahan: The 49ers have done almost everything possible under Shanahan except win a Super Bowl. Fair or not, not winning
a title with a stacked roster — especially on offense (Christian McCaffrey, Deebo Samuel, Brock Purdy, George Kittle) — would be a massive disappointment. His
job is in zero danger, but his rep may be.

SEATTLE SEAHAWKS | Quarterback Geno Smith: The Seahawks passed on drafting a QB in 2023 and re-signed Smith, but he has taken a significant step back. Entering
Thursday's game against Dallas, his completion percentage was down from 2022 (69.8 to 65.4 percent) despite the impressive offensive weapons around him. Seattle
has an out in his contract after this season that would only carry a $17.4 million dead-cap number, per Spotrac.
</item 22>

<item 23>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov. 17

Source: Daniel Kramer, MLB.com

Here’s the key piece of the article: “Industry sources told MLB.com this week that landing Ohtani doesn’t appear to be within the Mariners’ realistic agenda
this offseason.”

In a way, that’s surprising. The Mariners seem to check a lot of boxes for Ohtani. They’re set up to win for a long time with a bunch of young starting pitchers
and a co-superstar in Julio Rodriguez. Plus, Ohtani would get to be part of the quest for the franchise’s first World Series title, he’d get to wear the same
uniform as his baseball hero, Ichiro, and he’s lived there in the offseason. So it must be about the money, and Seattle’s reluctance to even get in the
conversation. That’s gotta be super disappointing for Mariners fans starved for a title, right?
</item 23>

<item 24>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
As for this season, not so much.

Sean Payton should be at the forefront of the conversation. It's one thing to turn around a team that was dreadful last season. But he has taken one that got
off to a horrible start and turned things around in-season.

Oh — and he has resurrected Russell Wilson's career.

That alone should put him at the top of the list, but he may need to get Denver into the playoffs to convince voters.

Comeback Player of the Year

Current favorite: Demar Hamlin

Longshot: Russell Wilson

Coming back from a life-threatening injury like a cardiac arrest makes Hamlin a great candidate to win CPOY. However, it would be understandable if voters were
to look elsewhere since Hamlin has seldom been active on game day and rarely seen the field.

It will depend on how vital voters deem playing to be.

Wilson was supposed to be the next great franchise quarterback for the Broncos when they acquired him prior to last season.
</item 24>

<item 25>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 25>

<item 26>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Stroud's Houdini act has helped him beat most of the defenses he's faced, and this one isn't worthy of being feared, but there is some worry Stroud have a
monster game without some massive yards after the catch (YAC) from his receivers. Fortunately he ranks ninth-best on the season among qualifying QBs in YAC per
completion (5.37).

STARTS: C.J. Stroud, Tank Dell, Nico Collins, Courtland Sutton, Devin Singletary, Javonte Williams, Russell Wilson

C.J. Stroud, Tank Dell, Nico Collins, Courtland Sutton, Devin Singletary, Javonte Williams, Russell Wilson SITS: Jerry Jeudy, both DSTs; Dalton Schultz is OUT

Start Him (Lineup Decisions) Projections powered by Sportsline Russell Wilson QB DEN Denver • #3

Age: 35 • Experience: 12 yrs.
</item 26>

<item 27>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 27>

<item 28>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.
</item 28>

<item 29>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Marshall prop bets at BetMGM Sportsbook

BetMGM Sportsbook offers some excellent player props for the Scooter's Coffee Frisco Bowl between the Roadrunners and Thundering Herd.

Frank Harris prop bets

237.5 passing yards OVER (-115) / UNDER (-115)

Passing TDs OVER 1.5 (-125) / UNDER 1.5 (-105)

Anytime touchdown scorer (-110)

Joshua Cephus props bets

80.5 receiving yards OVER (-115) / UNDER (-115)

Anytime touchdown scorer (-190)

Rasheen Ali prop bets

80.5 rushing yards OVER (-115) / UNDER (-115)

Anytime touchdown scorer (-175)

Kevorian Barnes prop bets

64.5 rushing yards OVER (-115) / UNDER (-115)

Anytime touchdown scorer (-135)

Robert Henry prop bets

Anytime touchdown scorer (-130)

UTSA vs.
</item 29>

<item 30>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Chase Harrison, QB (transfer portal)

Brandon McElroy, DL (transfer portal)

Caleb Coombs, WR (transfer portal)

Zion Odoms, WR (transfer portal)

Jayven Hall, LB (transfer portal)

Cam Fancher, QB (transfer portal)

Thomas Lane, DL (transfer portal)

Chinazo Obobi, OT (transfer portal)

Maurice Jones, RB (transfer portal)

Sean Sallis, TE (transfer portal)

Myles Bell, DB (transfer portal)

Trent Holler, iOL (transfer portal)

Joshua McTier, CB (transfer portal)

Jabarrek Hopkins, DL (transfer portal)

CK Obobi, OL (transfer portal)

Sean Ellis, TE (transfer portal)

TyShawn Hurst, OT (transfer portal)

Elijah Russell, EDGE (transfer portal)

UTSA vs.
</item 30>

<item 31>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Marshall prediction: Frank Harris leads Roadrunners past the Herd

Even with the transfer portal hitting this game hard, the UTSA offense will be too much for Marshall to keep up with. The Roadrunners push the pace in a way
that Marshall can't replicate, and the Thundering Herd aren't efficient enough to make the most of what few scoring opportunities they'll have.

The Thundering Herd could find themselves down several scores by the end of the first frame if Pennington turns the ball over, as he's prone to do. UTSA is a
well-oiled machine against teams with similar or worse talent, and Marshall is in that latter category.

It's a big number to lay, but we're on the favorite Tuesday night.

Pick: UTSA
</item 31>

<item 32>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.



He was emotional after defeating Penn State.



"I want to thank Coach Harbaugh. ... I love the s— out of you man."



🎥: @CFBONFOX pic.twitter.com/NmXZWrGxCM — The Sporting News (@sportingnews) November 11, 2023

Moore, however, will emerge as a leading candidate for some jobs as a result, and that is a huge victory for one of the nation’s top assistants.

Losers

Connor Stalions

Stalions made a name for himself. He will be remembered as the Michigan staffer as the center of this sign-stealing scandal. Stalions resigned on Nov. 3. The
NCAA investigation whether Michigan was involved with in-person scouting, which is prohibited by the NCAA, put Stalions into the college football consciousness.
</item 32>

<item 33>
Why is Brock Purdy called Mr. Irrelevant? Story behind 49ers QB's nickname explained (Sportskeeda, 2023-10-10T22:30:20+00:00)
Coral left the franchise ahead of his sophomore season.

5. Bailey Zappe, New England Patriots, Pick 137 - C-

Mac Jones' backup plays for the worst Patriots team of the past twenty years. It's not a dream spot for the Western Kentucky alum.

6. Sam Howell, Washington Commanders, Pick 144 - B

Howell is the franchise quarterback for the usually trigger-happy Washington Commanders franchise. This shows that the Commanders' new ownership group has faith
in Howell's ability to quarterback the team. He gets a positive grade for that sole reason.

7. Chris Oladokun, Pittsburgh Steelers, Pick 241 - F

Oladokun is yet to make a start in the NFL. He'll be rated an F until he changes that. However, that won't be on the team that drafted him, as he was waived by
the Steelers a while ago.

8.
</item 33>

<item 34>
Why is Brock Purdy called Mr. Irrelevant? Story behind 49ers QB's nickname explained (Sportskeeda, 2023-10-10T22:30:20+00:00)
He hasn't been spectacular this season, but at least he's starting games at the highest level.

3. Malik Willis, Tennessee Titans, Pick 86 - D

Willis is the QB2 or QB3 in Tennessee, depending on who you ask. That's interesting, considering that some draft analysts thought him the most naturally gifted
QB in the 2022 Draft class. Unfortunately, he is fighting a losing battle in Tennessee, as the franchise drafted the highly rated Will Levis to take his place.
A "D" grade might even be generous.

4. Matt Corral, Carolina Panthers, Pick 94 - E

The Panthers have mainly ignored Coral as they elected to trade up to number one for Bryce Young in the 2023 Draft. This tells you about the franchise's faith
in the Ole Miss alum. Coral left the franchise ahead of his sophomore season.

5.
</item 34>

<item 35>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 35>

<item 36>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
On the field, Michigan is in the midst of what could be a historic season. Off the field, the the focus has rarely been on the team's success.

An investigation into Michigan's alleged sign-stealing scheme, centered on now-fired staffer Connor Stalions, continues to evolve by the hour, with one update
after another adding another wrinkle to the saga that began in October. The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a
plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy
Valley.

The Sporting News is tracking the latest updates from the Michigan sign-stealing scandal. Follow along with the newest developments below.

MORE ON MICHIGAN SIGN-STEALING:

Michigan sign-stealing scandal updates

Friday, Nov.
</item 36>

<item 37>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Because we splurged for McCaffrey and Kamara, we had to sacrifice salary elsewhere, and London is a high-upside option even in an offense that doesn't throw the
ball with efficiency.

WR Jaxon Smith-Njigba, Seahawks @ 49ers ($4,100). Is a takeover underway in the Seahawks offense? Smith-Njigba is coming off a game against the Cowboys that saw
him collect seven-of-11 targets for 62 yards. It's a step up from a disappointing outing against San Francisco, but Tyler Lockett's involvement seems to be
lessening as of late with the rookie stepping up. It wasn't a great day for the Seattle offense on Thanksgiving against the Niners, but the Seahawks should put
up a better effort against a divisional opponent they know so well. If they're trailing, JSN should see action with a negative game script.
</item 37>

<item 38>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the season he's totaled 28 rushes for 134 yards and a score.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Sam Howell QB WAS Washington • #14

Age: 23 • Experience: 2 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS QB 31st PROJ PTS 21.3 QB RNK 14th YTD Stats PAYDS 961 RUYDS 82 TD 5 INT 5
FPTS/G 16.1 The Commanders like limiting Howell's pass attempts in general -- he's only thrown over 31 times in high-scoring shootouts like in Weeks 2 and 4.
While there's potential for that on Thursday, it's not a certainty. Howell could just as easily put up nominal stats to help the Commanders bring home a win.
</item 38>

<item 39>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

What are the biggest QB injuries in Week 14?

Leading into Week 13, Dorian Thompson-Robinson (concussion) was the lone QB injury situation we were closely monitoring. Even if Thompson-Robinson is able to
clear concussion protocol in time for Week 14, we expect Flacco to make his second start as a Brown.

This week, we unfortunately saw Kenny Pickett (ankle) and Derek Carr (head, shoulder, back) exit their respective contests with injuries, complicating their
statuses for Week 14. We've replaced both QBs with backups Mitchell Trubisky and Jameis Winston, respectively, as early indications point toward both missing
time.

UPDATE: Trevor Lawrence suffered an ankle injury on Monday, which will likely keep him out for at least a week and open the door for C.J. Beathard (vs. Browns)
to start.
</item 39>

<item 40>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 40>

<item 41>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Jourdan Lewis. JSN can mirror some of the production Curtis Samuel gave the Commanders last week in Dallas, but forget 100 yards. Around 50 yards seems like the
ceiling with a big boost from garbage time.

Sit: TEs Noah Fant and Will Dissly

Noah Gray is a better fantasy player than Noah Fant this season. Will Dissly is barely a better fantasy player than Will Mallory this season.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: Seahawks D/ST

The Seahawks are getting better defensively against the pass, but that's not enough to trust them here, as they're getting gashed on the ground, too.

Sit: K Jason Myers

This is actually not a good spot for one of the most reliable fantasy kickers, given the Cowboys don't give up many field-goal attempts.
</item 41>

<item 42>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 42>

<item 43>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Nobody has really emerged in the AFC to unseat these teams at the top.”

• Miami Dolphins: Has Tua Tagovailoa proven he’s built to last?

What I think: It was tough watching Tagovailoa last season without worrying for his physical well-being. I’ve never felt that way watching him so far this
season, and will feel even better if he plays a full game against Philadelphia in Week 7. New England, Kansas City, the Jets (twice), Washington, Tennessee,
Dallas, Baltimore and Buffalo will all get their shots at Tagovailoa. I’m cautiously optimistic, but think there will be games when defense muddy the picture
sufficiently to create turnovers.

Exec comment: “Interesting team. They put up 70 on Denver and should have put up 70 on the Giants. No one can keep up with that kind of firepower. But they also
haven’t really beaten anyone. They were down 14 points to Carolina.
</item 43>

<item 44>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Was this old-school Big Ten football or Big Ten West football in disguise?

Michigan's defense clamped on Penn State quarterback Drew Allar (10 of 22, 70 yards, TD), and the Nittany Lions had one play of more than 20 yards. Corum closed
the drama with a 30-yard TD.

The same-old questions about Nittany Lions coach James Franklin in big games resurfaced, but those are back-page headlines in the Big Ten East race. The quirky
tiebreaker involving Big Ten West opponents? That's out the door. The winner-take-all showdown between Ohio State and Michigan? There are a full two weeks of
anticipation for that now.

Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns. Moore's postgame emotion was a headline, but there is a
reality that set in through a brutal watch of a second half.

McCarthy is a leader.
</item 44>

<item 45>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 45>

<item 46>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 46>

<item 47>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
Pennington has thrown six interceptions and zero touchdowns on only 79 passing attempts this season.

Marshall's offense will continue to run through star RB Rasheen Ali. The senior speedster carried the offense even when Fancher was under center, totaling 1,256
yards and 15 touchdowns on the season. This offense has struggled despite Ali's efforts, ranking 93rd in scoring and 89th in yards per game.

On the season, Marshall is 4-8 ATS and 6-6 on the O/U board. The Herd won only three of their eight conference games on the season.

Marshall Thundering Herd opting out of Scooter's Coffee Frisco Bowl

While UTSA has the biggest individual loss for this game, the Herd's depth has taken a huge blow thanks to transfer portal entrants.
</item 47>

<item 48>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 48>

<item 49>
Why is Brock Purdy called Mr. Irrelevant? Story behind 49ers QB's nickname explained (Sportskeeda, 2023-10-10T22:30:20+00:00)
Eight quarterbacks were selected before Purdy in the 2023 NFL Draft. Their names are Kenny Pickett (Pittsburgh Steelers), Desmond Ridder (Atlanta Falcons),
Malik Willis (Tennessee Titans), Matt Corral (Carolina Panthers), Bailey Zappe (New England Patriots), Sam Howell (Washington Commanders), Chris Oladokun
(Pittsburgh Steelers) and Skylar Thompson (Miami Dolphins). As you can tell, the "Mr. Irrelevant" of this Draft has had, by far, the best NFL career so far from
the bunch.

Which way should Giants next year? Fire up our 2024 NFL Draft Simulator to get NY back in contention

The 2022 NFL Draft's Mr. Irrelevant is the franchise QB of one of the best teams in the NFL, the San Francisco 49ers. Since breaking into the starting lineup,
Purdy has made a mockery of his "Mr. Irrelevant" tag.
</item 49>

<item 50>
UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
The UTSA Roadrunners (8-4) take on the Marshall Thundering Herd (6-6) on Tuesday night in the Scooter's Coffee Frisco Bowl (9 p.m ET, ESPN). The matchup sees
the Roadrunners as large betting favorites on odds boards. Currently, USTA is favored by on the point spread, with Marshall available for on the moneyline.

The Roadrunners are in their 13th season as a program but have yet to win a bowl game. This is the fourth bowl game they've played in five years. Marshall is
competing in a bowl game for the seventh straight year.

These teams have been on different streaks as the year has progressed. UTSA has won seven of its last eight games, while Marshall has lost six of its last
eight.

Here are the live odds, team betting news, player props and our betting prediction for UTSA vs. Marshall in Tuesday's Scooter's Coffee Frisco Bowl.
</item 50>
=== END ITEM id=f40d129c5d769b5c ===

=== ITEM id=031ca8a5da19676e ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: A year has passed since Altman catapulted ChatGPT to global fame.
  Fact 0.1: Altman has become Silicon Valley's sought-after voice on the promise of artificial intelligence.
  Fact 0.2: Altman has become Silicon Valley's sought-after voice on the potential dangers of artificial intelligence.
  Fact 0.3: Altman's sudden exit was mostly unexplained.
  Fact 0.4: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: The piece covers this week's topics.
  Fact 1.1: Sam Altman is backing a teen's AI startup.
  Fact 1.2: Google held a hardware event.
  Fact 1.3: First impressions of the Pixel 8 Pro are covered.
  Fact 1.4: Flexport drama is covered.
  Fact 1.5: The ongoing FTX fallout is covered.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The board's language is a basis for the prevailing theory.
  Fact 2.1: The way giant tech companies work is a basis for the prevailing theory.
  Fact 2.2: The prevailing theory currently floating around is based on the board's language and how giant tech companies work.

Retrieved items:
<item 1>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Sam and I are shocked and saddened by what the board did today. Let us first say thank you to all the incredible people who we have worked with at OpenAI, our
customers, our investors, and all of those who have been reaching out. We too are still trying to figure out exactly… — Greg Brockman (@gdb) November 18, 2023

Murati told of Altman’s firing

Brockman alleges that Mira Murati, OpenAI’s CTO and now interim CEO, was informed on Thursday night that Altman would be fired.
</item 1>

<item 2>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Around the same time, Musk, who had co-chaired its board with Altman, resigned from the board in a move that OpenAI said would eliminate a “potential future
conflict for Elon” due to Tesla’s work on building self-driving systems.

While OpenAI’s board has preserved its nonprofit governance structure, the startup it oversees has increasingly sought to capitalize on its technology by
tailoring its popular chatbot to business customers.

At its first developer conference last week, Altman was the main speaker showcasing a vision for a future of AI agents that could help people with a variety of
tasks. Days later, he announced the company would have to pause new subscriptions to its premium version of ChatGPT because it had exceeded capacity.

Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
</item 2>

<item 3>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT Voice rolls out to all free users

Even if its leadership is in flux, OpenAI is still releasing updates to ChatGPT. First announced in September and granted to paid users on a rolling basis, the
text-to-speech model can create a voice from text prompts and a few seconds of speech samples. OpenAI worked with voice actors to create the five voice options,
and you can give it a shot by heading to the settings in your mobile ChatGPT apps and tapping the “headphones” icon.

Sam Altman might return, but it’s complicated

The only constant within OpenAI right now is change, and in a series of interviews, Nadella hedged on earlier reporting that Altman and Brockman were headed to
Microsoft.

“Obviously, we want Sam and Greg to have a fantastic home if they’re not going to be in OpenAI,” Nadella said in an interview with CNBC, saying that we was
“open” to them settling at Microsoft or returning to OpenAI should the board and employees support the move.
</item 3>

<item 4>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“This is another indication that AI is progressing faster than even many experts anticipated,” Victoria Krakovna, who would soon join DeepMind as an “AI safety”
researcher, wrote in a blog post. DeepMind’s founders were increasingly worried about what Google would do with their inventions. In 2017, they tried to break
away from the company. Google responded by increasing the salaries and stock award packages of the DeepMind founders and their staff. They stayed put. The
ethics board never had a second meeting. The Breakup

Convinced that Page’s optimistic view of AI was dead wrong, and angry at his loss of DeepMind, Musk built his own lab. OpenAI was founded in late 2015, just a
few months after he met with Altman at the Rosewood hotel in Silicon Valley. Sam Altman Credit: Bloomberg Businessweek Musk pumped money into the lab, and his
former PayPal buddies – Hoffman and Thiel – came along for the ride.
</item 4>

<item 5>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 5>

<item 6>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
The two have since stopped speaking. Elon Musk and Sam Altman. Credit: Bloomberg, AP “There is disagreement, mistrust, egos,” Altman said. “The closer people
are to being pointed in the same direction, the more contentious the disagreements are. You see this in sects and religious orders. There are bitter fights
between the closest people.” Last month that infighting came to OpenAI’s boardroom. Rebel board members tried to force out Altman because, they believed, they
could no longer trust him to build AI that would benefit humanity. Over five chaotic days, OpenAI looked as if it were going to fall apart, until the board –
pressured by giant investors and employees who threatened to follow Altman out the door – backed down.

The drama inside OpenAI gave the world its first glimpse of the bitter feuds among those who will determine the future of AI.
</item 6>

<item 7>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a memo internally, he published the text on X.

After learning today’s news, this is the message I sent to the OpenAI team: https://t.co/NMnG16yFmm pic.twitter.com/8x39P0ejOM — Greg Brockman (@gdb) November
18, 2023

Senior OpenAI researchers resign

Three senior OpenAI researchers resign after Brockman, including the director of research Jakub Pachocki and head of preparedness Aleksander Madry.

—

November 16

Ilya Sutskever schedules call with Altman

According to a post on X (formerly Twitter) from Brockman, Ilya Sutskever, the chief scientist at OpenAI and a co-founder, texted Altman on Thursday evening
about scheduling a Friday noon call.

Sam and I are shocked and saddened by what the board did today.
</item 7>

<item 8>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Could Altman have been working with — or against — OpenAI’s patron in secret? If Altman wanted to kill the golden goose by going independent, that might have
activated the board’s fiduciary or otherwise statutory duty. On the other hand, if he was negotiating some other deal, like an acquisition or deeper and more
exclusive integration, it could also have caused the board to bristle, either at the idea itself or at being excluded.

But if Microsoft was as shocked as the rest of us, as one report has it, it could hardly be the kind of high-stakes conspiracy some seem to be hoping for. But
one must assume that Microsoft would say that either way. Even if they’d been working with Altman on some kind of secret plan, they can truthfully say they were
surprised by his firing. (And they “remain committed to our partnership.”)

2. Do they disagree on long-term strategy?
</item 8>

<item 9>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 9>

<item 10>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Musk and Page stopped speaking soon after the party that summer. A few weeks later, Musk dined with Altman, who was then running a tech incubator, and several
researchers in a private room at the Rosewood hotel in Menlo Park, California, a favoured deal-making spot close to the venture capital offices of Sand Hill
Road.

That dinner led to the creation of a startup called OpenAI later in the year. Backed by hundreds of millions of dollars from Musk and other funders, the lab
promised to protect the world from Page’s vision. Thanks to its ChatGPT chatbot, OpenAI has fundamentally changed the technology industry and has introduced the
world to the risks and potential of artificial intelligence. OpenAI is valued at more than $US80 billion ($A123 billion), according to two people familiar with
the company’s latest funding round, although Musk and Altman’s partnership didn’t make it. The two have since stopped speaking. Elon Musk and Sam Altman.
</item 10>

<item 11>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
In a post on X (formerly Twitter), Altman wrote that he “loved his time at OpenAI” and will have “more to say about what’s next later.”

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

OpenAI’s dramatic leadership change will also see Greg Brockman — who was a member of the team that co-founded OpenAI, as was Altman — step down as chairman of
the board but remain as OpenAI’s president, reporting to the company’s newly-appointed interim CEO, Mira Murati. Murati was previously OpenAI’s CTO.

OpenAI says that it’ll begin conducting a formal search for a permanent CEO immediately.
</item 11>

<item 12>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 12>

<item 13>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
The company said its board consists of OpenAI’s chief scientist, Ilya Sutskever, and three non-employees: Quora CEO Adam D’Angelo, tech entrepreneur Tasha
McCauley, and Helen Toner of the Georgetown Center for Security and Emerging Technology.

OpenAI’s key business partner, Microsoft, which has invested billions of dollars into the startup and helped provide the computing power to run its AI systems,
said that the transition won’t affect its relationship.

“We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of AI to our customers,” said an
emailed Microsoft statement.

While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s. He was recruited in 2014 to take lead of
the startup incubator YCombinator.
</item 13>

<item 14>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman posted Friday on X: “i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved
working with such talented people. will have more to say about what’s next later.”

The Associated Press and OpenAI have a licensing and technology agreement allowing OpenAI access to part of the AP’s text archives.

Altman helped start OpenAI as a nonprofit research laboratory in 2015. But it was ChatGPT’s explosion into public consciousness that thrust Altman into the
spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media. On a world tour this year, he was mobbed by
a crowd of adoring fans at an event in London.

He’s sat with multiple heads of state to discuss AI’s potential and perils.
</item 14>

<item 15>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
one takeaway: go tell your friends how great you think they are. — Sam Altman (@sama) November 18, 2023

Investors pushing for Altman’s return

Investors — furious at the turn of events — are reportedly exerting pressure on OpenAI’s board to reinstate Altman, going so far as to recruit Microsoft.
Nadella is said to be sympathetic.

Board agrees to reverse course — in principle

The Verge reports that the board agreed in principle to resign and to allow Altman and Brockman to return. It waffled, however, missing a deadline yesterday by
which many OpenAI staffers were set to leave the company. Altman is said to be ambivalent about coming back and asking for “significant” governance changes.

—

November 17

Brockman demoted

Brockman says he got a text from Sutskever shortly after noon on Friday asking for a quick call.
</item 15>

<item 16>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 16>

<item 17>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a Google Meet link, Brockman was told that he was being removed from the board as chairman “but was vital to the company and would retain his
role” as president, and that Altman had been fired.

Altman’s firing publicly announced

OpenAI published a post on its blog announcing the executive shake-up. The company’s management team was aware shortly after.

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

All-hands meeting

OpenAI held an all-hands meeting Friday afternoon during which Sutskever defended Altman’s ouster. He dismissed suggestions that pushing Altman out amounted to
a “hostile takeover,” and claimed that it was necessary to protect OpenAI’s mission of “making AI beneficial to humanity.”
</item 17>

<item 18>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman also co-founded Worldcoin, a biometric and cryptocurrency project that’s been scanning people’s eyeballs with the goal of creating a vast digital
identity and financial network.

___

Associated Press business writers Haleluya Hadero in New York, Kelvin Chan in London and Michael Liedtke and David Hamilton in San Francisco contributed to this
report.
</item 18>

<item 19>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 19>

<item 20>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
2. Do they disagree on long-term strategy?

Despite being the hottest tech company in the world right now and everyone talking about ChatGPT, OpenAI isn’t exactly a sound business. It’s shoveling money
into the furnace as fast as it can by serving, by all accounts, a fantastically expensive product at bargain-bin prices.

That’s all well and good for a year or two, but at some point that strategy changes from a growth hack to an existential liability. Could Altman and the board
have had irreconcilable differences on where that point lies?

This doesn’t seem so likely. The company has been very deliberately pursuing this very publicly, confidently and on a long-term basis. Altman and the board seem
to be in sync on this, at least for the present.

3. Do the numbers not add up?

On the other hand, could OpenAI be losing even more money than Altman admitted or projected? It seems impossible, but the costs of running this operation have
no precedent, nor really does the operation itself.
</item 20>

<item 21>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
In a dramatic turn of events late Friday, ex-Y Combinator president Sam Altman was fired as CEO of AI startup OpenAI, the company behind viral AI hits like
ChatGPT, GPT-4 and DALL-E 3, by OpenAI’s board of directors. Then, the company’s longtime president and co-founder, Greg Brockman, resigned — as did three
senior OpenAI researchers. And the fallout continues.

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

It’s a fast-moving situation that we’re still trying to get to the bottom of. No doubt more will become clear as time goes on. To make it easier to follow all
that’s happened in the meantime, though, we’ve put together a timeline; we’ll do our best to keep it current.
</item 21>

<item 22>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Microsoft releases a statement

Satya Nadella, the CEO of Microsoft, a major investor in — and partner with — OpenAI, published a statement about Altman’s firing:

“As you saw at Microsoft Ignite this week, we’re continuing to rapidly innovate for this era of AI, with over 100 announcements across the full tech stack from
AI systems, models and tools in Azure, to Copilot. Most importantly, we’re committed to delivering all of this to our customers while building for the future.
We have a long-term agreement with OpenAI with full access to everything we need to deliver on our innovation agenda and an exciting product roadmap; and remain
committed to our partnership, and to Mira and the team. Together, we will continue to deliver the meaningful benefits of this technology to the world.”

Brockman quits

Brockman announced his resignation from OpenAI, citing “today’s news.”
</item 22>

<item 23>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
OpenAI says that it’ll begin conducting a formal search for a permanent CEO immediately.

“OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence benefits all humanity,” the board wrote in a joint
statement. “The board remains fully committed to serving this mission. We are grateful for Sam’s many contributions to the founding and growth of OpenAI. At the
same time, we believe new leadership is necessary as we move forward. As the leader of the company’s research, product, and safety functions, Mira is
exceptionally qualified to step into the role of interim CEO. We have the utmost confidence in her ability to lead OpenAI during this transition period.”

Sam Altman is a hero of mine. He built a company from nothing to $90 Billion in value, and changed our collective world forever. I can't wait to see what he
does next. I, and billions of people, will benefit from his future work- it's going to be simply incredible.
</item 23>

<item 24>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 24>

<item 25>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 25>

<item 26>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 26>

<item 27>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 27>

<item 28>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 28>

<item 29>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
—

November 18

“Not … in response to malfeasance”

In an internal memo obtained by Axios sent Saturday morning, OpenAI COO Brad Lightcap said yesterday’s announcement “took [the management team] by surprise” and
that management had had “multiple conversations with the board to try to better understand the reasons and process behind their decision.” Discussions were
ongoing as of Saturday morning, per the memo.

“We can say definitively that the board’s decision was not made in response to malfeasance or anything related to our financial, business, safety, or
security/privacy practices,” Lightcap added. “This was a breakdown in communication between Sam and the board … We still share your concerns about how the
process has been handled, are working to resolve the situation, and will provide updates as we’re able.”

OpenAI’s funding in jeopardy

The planned sale of OpenAI employee shares that would value the startup at about $86 billion could be in jeopardy.
</item 29>

<item 30>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Thank you… — Eric Schmidt (@ericschmidt) November 17, 2023

OpenAI’s board of directors now consists of OpenAI’s chief scientist Ilya Sutskever; Quora CEO Adam D’Angelo; tech entrepreneur Tasha McCauley; and Helen Toner,
the director of strategy at Georgetown University’s Center for Security and Emerging Technology.

Altman’s removal comes unexpectedly, to say the least. The company just last week hosted its first developer conference, OpenAI DevDay, which Altman emceed.
Altman spoke at the Asia-Pacific Economic Cooperation conference and an Oakland, California event on Thursday. And according to The Verge and The New York
Times, citing multiple internal sources, OpenAI employees learned of Altman’s firing when it was announced publicly.

Altman has a long history steering OpenAI.
</item 30>

<item 31>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Brockman was invited to join — but it’s unclear whether he’ll take execs up on that invitation.

Board negotiations hit a snag

Bloomberg reports that Lightcap and Murati, among others, are pushing the board to reinstate Altman. But unsurprisingly, the directors are resisting. As of
midday Sunday, the board hadn’t resigned out of concern over who could replace them, and were vetting candidates. One possible new addition could be Salesforce
co-CEO Bret Taylor.

Altman out, Shear in

Altman won’t be returning as CEO, according to a report in The Information citing an internal memo sent by Sutskever. As the search for a new permanent CEO
continues, OpenAI has appointed Emmett Shear, the co-founder of video streaming site Twitch, as interim CEO — replacing Murati.
</item 31>

<item 32>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
It has been a whirlwind four days for OpenAI, the generative AI poster child behind the smash hit ChatGPT.

Seemingly out of nowhere, the OpenAI board ousted CEO and co-founder Sam Altman and demoted president and co-founder Greg Brockman, who subsequently resigned,
paving the way for what looked like a mutiny by staff insisting the founders be reinstated post-haste. By then, Microsoft had already hired Altman and Brockman
to head up a new internal AI unit, though, as things transpired, nothing had actually been signed yet, with rumors suggesting that the ousted leaders might
actually return to OpenAI after all — in some capacity, at least.

The situation remains fluid, and any number of potential outcomes still remain on the table. But the whole debacle has shone a spotlight on the forces that
control the burgeoning AI revolution, leading many to question what happens if you go all-in on a centralized proprietary player, and what happens if things
then go belly-up?
</item 32>

<item 33>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He said OpenAI still has a “deep bench of technical leaders” but its next executives will have to steer it through the challenges of scaling the business and
meeting the expectations of regulators and society.

Forrester analyst Rowan Curran speculated that Altman’s departure, “while sudden,” did not likely reflect deeper business problems.

“This seems to be a case of an executive transition that was about issues with the individual in question, and not with the underlying technology or business,”
Curran said.

Altman has a number of possible next steps. Even while running OpenAI, he placed large bets on several other ambitious projects.

Among them are Helion Energy, for developing fusion reactors that could produce prodigious amounts of energy from the hydrogen in seawater, and Retro
Biosciences, which aims to add 10 years to the human lifespan using biotechnology.
</item 33>

<item 34>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Confirmation Sam Altman will not return as OpenAI’s CEO

A number of investors and OpenAI employees tried to bring back Altman after his sudden firing by the company’s board, but following a weekend of negotiations,
it was confirmed that Altman would not return to OpenAI and new leadership would take hold. What this means for ChatGPT’s future, and for the OpenAI Dev Day
announcements, remains to be seen.

Sam Altman ousted as OpenAI’s CEO

Sam Altman has been fired from OpenAI. He will leave the company’s board and step down as CEO, with OpenAI’s chief technology officer Mira Murati stepping in as
interim CEO. In a blog post from OpenAI, the company writes that the board “no longer has confidence in [Altman’s] ability to continue leading OpenAI.”

In a statement on X, Altman said working at OpenAI “was transformative” for him and “hopefully the world.”
</item 34>

<item 35>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Or what if, and again this is purely speculation, Altman has been secretly pursuing an internal project, perhaps at significant cost, against the advice of the
board and without the necessary safety measures that probably should accompany such research? It sounds a little wild, but firing your CEO like this is also a
little wild.

Some kind of major mismatch in the financial department could be cause for dismissal, but it’s hard to imagine what Altman could have kept from the board and
CTO that would be so damning.

There’s also the possibility that Altman was making personal investments in a way that the board disagreed with. With OpenAI poised to be a kingmaker in the
field, he would certainly be in a position of power. One would think that, as an ideologically driven person already rich beyond belief and at the head of the
world’s leading AI company, Altman would have risen beyond having to do this kind of side deal, or at the very least that scrutiny on him and those close to him
would prevent them. But one can never be sure.

4.
</item 35>

<item 36>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Researchers inside OpenAI, Google and other companies thought this rapidly improving technology could be a path to AGI. But Amodei was unhappy about the
Microsoft deal because he thought it was taking OpenAI in a really commercial direction. He and other researchers went to the board to try to push Altman out,
according to five people familiar with the matter. After they failed, they left. Like DeepMind’s founders before them, they worried that their new corporate
overlords would favour commercial interests over safety.

In 2021, the group of about 15 engineers and scientists created a new lab called Anthropic. The plan was to build AI the way the effective altruists thought it
should done – with very tight controls. “There was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic,” said an Anthropic spokesperson,
Sally Aldous.
</item 36>

<item 37>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.
</item 37>

<item 38>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
But later on X, formerly Twitter, Brockman posted a message he sent to OpenAI employees in which he wrote, “based on today’s news, i quit.”

In another X post on Friday night, Brockman said Altman was asked to join a video meeting at noon Friday with the company’s board members, minus Brockman,
during which OpenAI co-founder and Chief Scientist Ilya Sutskever informed Altman he was being fired.

“Sam and I are shocked and saddened by what the board did today,” Brockman wrote, adding that he was informed of his removal from the board in a separate call
with Sutskever a short time later.

OpenAI declined to answer questions on what Altman’s alleged lack of candor was about. The statement said his behavior was hindering the board’s ability to
exercise its responsibilities.

Altman posted Friday on X: “i loved my time at openai.
</item 38>

<item 39>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Sam Altman has been fired from OpenAI, Inc., the 501(c)(3) nonprofit that acts as the governing body for OpenAI, the AI startup behind ChatGPT, DALL-E 3, GPT-4
and other highly capable generative AI systems. He’ll both leave the company’s board of directors and step down as CEO.

In a post on OpenAI’s official blog, the company writes that Altman’s departure follows a “deliberative review process by the board” that concluded that Altman
“wasn’t consistently candid in his communications” with other board members, “hindering its ability to exercise its responsibilities.”

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

“The board no longer has confidence in [Altman’s] ability to continue leading OpenAI,” the blog post reads.
</item 39>

<item 40>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.
</item 40>

<item 41>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
ChatGPT-maker Open AI said Friday it has pushed out its co-founder and CEO Sam Altman after a review found he was “not consistently candid in his
communications” with the board of directors.

“The board no longer has confidence in his ability to continue leading OpenAI,” the artificial intelligence company said in a statement.

In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial
intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.

Mira Murati, OpenAI’s chief technology officer, will take over as interim CEO effective immediately, the company said, while it searches for a permanent
replacement.

The announcement also said another OpenAI co-founder and top executive, Greg Brockman, the board’s chairman, would step down from that role but remain at the
company, where he serves as president.
</item 41>

<item 42>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 42>

<item 43>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Microsoft has poured $13 billion into OpenAI so far and has what amounts to a 49% stake in the company. The former’s stock price fell more than 1% in the last
30 minutes of trading, after Altman’s departure was announced; Axios reports Microsoft was notified Altman would be leaving “minutes” before the public
announcement.

OpenAI was founded as a nonprofit in 2015, but restructured in 2019 as a “capped-profit” company to raise capital — an acknowledgement of the huge costs
associated with training cutting-edge AI systems. Underlining the point, Altman said in an interview this week with the Financial Times that he “hoped”
Microsoft would increase its investment to help pay for “huge” imminent model training costs.

When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post.
</item 43>

<item 44>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 44>

<item 45>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Any significant snafu in that department would reflect on her, as well as Altman.

5. Perhaps a difference of AI ethics or philosophy?

Altman is a proud techno-optimist, and often speaks fondly of the possibilities of AGI, or artificial general intelligence, a theoretical software system that
achieves human-like intellect and versatility.

The board’s statement pointedly includes that “OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence
benefits all humanity” and that new leadership was necessary. It’s possible that Sam’s zeal for AGI, even absent a secret project or agreement, led to a major
rift between him and the board.

It’s been obvious to all that Altman took the company in a much more corporate direction from its origins, changing its legal status and aggressively pursuing
enterprise and consumer applications. That doesn’t sound a lot like the “mission” the board wants to advance.
</item 45>

<item 46>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Nadella leaves the door open to other OpenAI staffers, saying that they’ll be given the resources they need should they choose to join.

Sutskever’s mea culpa

Sutskever publishes a post on X suggesting that he regrets his decision to remove Altman and that he’ll do everything in his power to reinstate Altman as CEO.

I deeply regret my participation in the board's actions. I never intended to harm OpenAI. I love everything we've built together and I will do everything I can
to reunite the company. — Ilya Sutskever (@ilyasut) November 20, 2023

Employees threaten to resign

Nearly 500 of OpenAI’s roughly 770 employees — including, remarkably, Sutskever — publish a letter saying that they might quit unless the startup’s board
resigns and reappoints the ousted Altman. Later Monday, that number climbed to over 650.
</item 46>

<item 47>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI CEO Sam Altman had previously pointed to industry-wide GPU shortages for the service’s spotty performance. Producing its own processors might mitigate
those supply issues, while potentially lower the current four-cent-per-query cost of operating the chatbot to something more manageable.

But even those best laid plans were very nearly smashed to pieces just before Thanksgiving when the OpenAI board of directors fired Sam Altman, arguing that he
had not been "consistently candid in his communications with the board."

That firing didn't take. Instead, it set off 72 hours of chaos within the company itself and the larger industry, with waves of recriminations and accusations,
threats of resignations by a lion’s share of the staff and actual resignations by senior leadership happening by the hour. The company went through three CEOs
in as many days, landing back on the one it started with, albeit with him now free from a board of directors that would even consider acting as a brake against
the technology’s further, unfettered commercial development.
</item 47>

<item 48>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 48>

<item 49>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
But one can never be sure.

4. Could it be a major security or privacy incident?

The idea that the company has experienced a major, perhaps pervasive, security issue is bolstered by the fact that Microsoft reportedly suspended use of ChatGPT
internally a few days ago. OpenAI subsequently stopped allowing new signups. If there was a serious security problem in its biggest product and Altman
downplayed it, that would obviously create distrust with the board.

There is also the potential for misuse at scale with the enormous amount of personal data that travels through OpenAI’s APIs and services.

Working against this theory is the fact that CTO Mira Murati was just elevated to interim CEO in Altman’s place. It seems unlikely that anything
security-related would go through the CEO and not the CTO, or that the two would be at odds to the point where one could be fired like this and the other
swapped in to clean up the mess. As the board’s statement notes, Murati is in charge of product and safety, among other things.
</item 49>

<item 50>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Couldn't be a prouder brother. — Jack Altman (@jaltma) November 17, 2023

It’s unclear what missteps Altman might’ve made helming OpenAI… at present. But they evidently relate to his relationship with OpenAI’s rather unusual board
makeup and corporate governance structure — and perhaps OpenAI’s active talks to raise substantial new capital.

As a recent piece in VentureBeat explored, OpenAI’s for-profit subsidiary, OpenAI Global, LLC, is fully controlled by OpenAI’s nonprofit. While the for-profit
subsidiary is permitted to commercialize its tech, it’s subject to the nonprofit’s mission: attaining artificial general intelligence (AGI), or AI that can — as
OpenAI defines it — “outperform humans at most economically valuable work.”

The nonprofit OpenAI’s board has the power to both determine when the company has achieved AGI and exclude this AGI from IP licenses and other commercial terms,
including with Microsoft, one of OpenAI’s biggest investors and a steadfast integrator of OpenAI’s various technologies.
</item 50>
=== END ITEM id=031ca8a5da19676e ===
