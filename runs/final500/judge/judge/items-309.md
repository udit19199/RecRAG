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

=== ITEM id=48e540061fa3189a ===
Gold sentences and facts:
Sentence 0: If a piece of news or a prevailing sentiment starts to gain traction about a particular team's increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.
  Fact 0.0: News or a prevailing sentiment can start to gain traction about a team's increased chances of winning.
  Fact 0.1: Such traction can influence how people bet.
  Fact 0.2: This can lead sportsbooks to adjust their lines accordingly.
Sentence 1: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability.
  Fact 1.0: A significant amount of money may be placed on one team or participant.
  Fact 1.1: Sportsbooks might adjust the odds to balance their liability.
Sentence 2: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 2.0: Every sportsbook follows when a line moves or odds change for a legitimate reason.
  Fact 2.1: An injury or a roster move is an example of a legitimate reason for a line or odds change.
Sentence 3: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 3.0: Sportsbooks are also gathering this information.
  Fact 3.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.

Retrieved items:
<item 1>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Do point spread odds change?

Yes, point spread odds can change, and these shifts are commonly referred to as "line movement."

Lines move when the point spread or the odds associated with a particular bet change after sportsbooks have initially set them. There are several reasons for
these adjustments.

One of the most common causes is an imbalance in the amount of money being wagered on one side of the bet. If a significant majority of bettors are placing
their money on one team to cover the spread, the top sports betting apps will adjust the point spread to make the other side more appealing, thereby balancing
the betting action.

Another factor that can affect the spread odds is the occurrence of unexpected events, such as injuries to key players, changes in team strategies, or even
significant weather events that might impact a game's outcome.

For instance, if a star quarterback is announced as injured just days before a big game, the point spread might shift in favor of the opposing team.

Moreover, information and public perception play a role.
</item 1>

<item 2>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 2>

<item 3>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
See more: Best NCAA basketball betting sites

2023-24 NBA Season

October 24 Regular season begins November 3 - December 9 In-season tournament February 16-18 All-Star Weekend 2024 April 14 Regular season ends April 16-19
Play-in tournament April 20 Playoffs begin June 6 NBA Finals Game 1

NBA spreads, money lines, and over/unders

For anyone new to NBA betting, the staple wager types are:

NBA point spreads: Bet on games via margins of victory or loss. Say the New York Knicks are -7.5 favorites over the Miami Heat. That means the Knicks need to
win the game by 8 points to win a spread bet. On the other hand, the Heat can lose by up to 7 points (or win outright).

NBA moneylines: The simplest form of betting. You pick the winner of the game, and if you’re right, you’ll get paid out based on the odds when you finalize your
bet slip.
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
</item 5>

<item 6>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Moreover, information and public perception play a role. If a piece of news or a prevailing sentiment starts to gain traction about a particular team's
increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.

Point spread odds can and do change based on a combination of actual betting patterns, unexpected events, and shifts in public perception. Savvy bettors keep a
close eye on these line moves, as they can offer insights into where the value lies in a particular bet.
</item 6>

<item 7>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
This includes your initial $10 wager, along with a profit of $15.30.

It's essential to note that this calculation gives you the total payout, not just your earnings. If you're keen on tracking your profits from betting, remember
to subtract your original stake from the payout to get an accurate figure.

Why do moneyline odds change?

Lines at the best sports betting apps change quickly due to two primary reasons:

Public betting

New information

Before setting odds for an event, oddsmakers gather extensive data. Once they finalize the odds, they release them to the public.

As a bettor, you can bet immediately or wait for potential shifts in the odds, although predicting these shifts is challenging.

For instance, when examining odds for the next Super Bowl champion released shortly after the previous Super Bowl, these odds are based mostly on the recently
concluded season.

As the new season approaches and teams make changes, odds adjust based on perceived championship chances. As games progress, lines continue to evolve based on
team performances.

Public betting can also sway lines.
</item 7>

<item 8>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=48e540061fa3189a ===

=== ITEM id=61a62b5db092596d ===
Gold sentences and facts:
Sentence 0: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 0.0: The trial's standout testimony in the second week came from Caroline Ellison.
  Fact 0.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 0.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 0.3: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 0.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 0.5: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 1: Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).
  Fact 1.0: Bankman-Fried was even compared to Warren Buffet.
  Fact 1.1: Many people called Bankman-Fried the white horse of crypto.
  Fact 1.2: TechCrunch never called Bankman-Fried the white horse of crypto.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth.
  Fact 2.2: The prosecution said Bankman-Fried committed fraud to achieve great power.
  Fact 2.3: The prosecution said Bankman-Fried committed fraud to achieve great influence.
  Fact 2.4: The defense countered that the FTX founder acted in good faith.
  Fact 2.5: The defense said the FTX founder never meant to commit fraud.
  Fact 2.6: The defense said the FTX founder never meant to steal.
  Fact 2.7: The defense said the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 1>

<item 2>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 4>

<item 5>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 5>

<item 6>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 6>

<item 7>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 9>

<item 10>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”
</item 10>
=== END ITEM id=61a62b5db092596d ===

=== ITEM id=f3feb308ae2445be ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 0.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 0.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.
Sentence 1: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 1.0: Google unveiled new hardware.
  Fact 1.1: This week was Google's annual hardware event.
  Fact 1.2: At the event, Google showed off what it has been working on.
  Fact 1.3: Google is a search and consumer tech giant.

Retrieved items:
<item 1>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 1>

<item 2>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 9>

<item 10>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Not Sonos fast there: Audio company Sonos scored a big $32.5 million win against Google a while back. Now it transpires that the company’s patents were deemed
unenforceable and invalid. Whoops. A federal judge threw out the $32.5 million win this week.

When it clicks, it really clicks: We took a deep look at Pixel 8, and our team discovered that it delivers solid performance and design upgrades. The camera got
a particularly enthusiastic shout-out, with Darrell declaring that Google’s Pixel 8 Pro camera is the new mobile photography champ, and Brian waxing lyrical
about the phone’s picture-snapper in “The camera’s still the thing.”

Who needs computers anyway?: It seems like all our mobile devices may be starting to cannibalize sales of personal computers — Ron reports that PC shipments
decline slows in Q3 2023 and that Apple’s sales plunge over 23%.

Let’s get together

Reddit’s API-powered chaos continues, but it appears that things are starting to resolve a little.
</item 10>
=== END ITEM id=f3feb308ae2445be ===

=== ITEM id=d4661944c14f4271 ===
Gold sentences and facts:
Sentence 0: While his Republican colleagues have largely shunned TikTok, Ramaswamy is presenting himself as one of the Cool Politicians who will actually use it.
  Fact 0.0: Ramaswamy's Republican colleagues have largely shunned TikTok.
  Fact 0.1: Ramaswamy is presenting himself as one of the Cool Politicians.
  Fact 0.2: The Cool Politicians are politicians who will actually use TikTok.
Sentence 1: Artists have spent the last few years trying to drum up engagement for their work by making their songs trend on TikTok, which audiences have started to resist.
  Fact 1.0: Artists have spent the last few years trying to drum up engagement for their work by making their songs trend on TikTok.
  Fact 1.1: Audiences have started to resist artists making their songs trend on TikTok.

Retrieved items:
<item 1>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Republicans still don’t know how to talk to young voters online Vivek Ramaswamy struggles to gain an edge in the face of TikTok trolls

In an appeal to younger voters, Republican presidential candidate Vivek Ramaswamy — who proposed raising the voting age to 25 — launched his TikTok presence
with an endorsement from Jake Paul, the YouTuber turned boxer who built a content empire by marketing to children.

Ramaswamy is one of the only Republican politicians making an effort to connect with Gen Z and young millennials, a demographic that overwhelmingly supported
Democrats in the midterm elections. Despite the popularity and growing influence of far-right creators online, Republican candidates have historically failed to
engage young voters on social media, if they try at all.

While Democratic politicians like Rep. Alexandria Ocasio-Cortez and Sen. John Fetterman’s social media campaign strategies set the blueprint for politicians
seeking internet fame, Republicans have neglected to adopt the digital fluency required to secure the youth vote.
</item 1>

<item 2>
Republican candidates lag Democrats in engaging young voters online: in 2022, TikTok accounts were far more common among Democratic Senate and House candidates.
Vivek Ramaswamy’s TikTok launch, backed by Jake Paul, drew views but also trolling, amid contradictions between courting youth and calling TikTok dangerous and
proposing to restrict voting under 25. Experts say Gen Z values authenticity, platform fluency and credible creators—not forced trends or transactional
endorsements. Democratic successes include AOC’s Twitch streams
</item 2>

<item 3>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“Gen Z does not put up with bullshit,” Henry said. “That’s so much of what platforms like TikTok and Twitch cater to, and why they thrive on those platforms,
because you can just pull out your phone and be talking while you’re getting ready with the worst angle possible. Republicans in general, it goes against so
much of how they act otherwise and young people know that.”

Ramaswamy’s TikTok strategy is puzzling; he appears to be trying to replicate the success of previous candidates who became social media stars over the course
of their campaigns, but his TikTok presence conflicts with his own stances on social media and young voters. His communications director, Tricia McLaughlin, did
not immediately respond to TechCrunch’s request for comment.

Like many of his Republican colleagues, Ramaswamy has accused TikTok of being a threat to national security, and during a town hall days before he launched his
account, described the platform as “digital fentanyl” from China.
</item 3>

<item 4>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 4>

<item 5>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
A similar pattern was found in the four other countries they study looked at.

The study also found that the TikTok audience can be an effective marketing tool for artists, as TikTok users, in particular, are considerably more likely to
discover and share music than overall social media and short video platform (SFV) users.

In the UK, TikTok users were 108% more likely to discover and share music than other social media users, while in Germany, they were 100% more likely. US TikTok
users were 71% more likely to discover and share music.

“In doing so, TikTok users help both emerging and established artists to reach new audiences,” the Luminate report stated.

The study found that TikTok’s platform appears to be a major driver of the glocalization of music – the trend of local artists, performing in local languages,
being elevated to a regional or global platform.

“TikTok users have a stronger preference for international music than the average music listener,” the report said.
</item 5>

<item 6>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Only the occasional tweets. Meet @Vivek Ramaswamy ♬ original sound – GenosPicks

Establishing himself as the millennial politician who’s cool enough to use TikTok but anti-woke enough to play in the divisive culture war isn’t working in
Ramaswamy’s favor, though.

His account, which has amassed over 50,000 followers in the weeks since he joined, has been barraged with comments either criticizing his positions or trolling
him. He’s been the butt of Gen Z’s relentless comments about getting off to his content (the top comments on his videos are consistently jokes about edging). He
also incited the wrath of witchtok creators, who filmed themselves casting hexes upon him and other conservatives.

Ramaswamy is raking in TikTok engagement — even if it’s not how he intended — but Democrats are still dominating the Republican party in digital strategy.
Ocasio-Cortez’s Instagram Stories garnered her nationwide popularity when she campaigned in 2018, reaching further than her local constituency.
</item 6>

<item 7>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
MBW’s Stat Of The Week is a series in which we highlight a data point that deserves the attention of the global music industry. Stat Of the Week is supported by
Cinq Music Group , a technology-driven record label, distribution, and rights management company.

TikTok is in the process of reinventing itself as both a partner to music companies, and a competitor to streaming and distribution services.

Now, a new market analysis sheds some light on why the ByteDance-owned short video platform has focused so intensely on expanding its services in the music
world, with the firm reporting to be a major driver of music consumption and spending.

While that may not come as a surprise to those who’ve noted the overlap between TikTok and music culture, the numbers in the analysis shine a light on the scale
of TikTok’s influence.

According to the analysis, commissioned by TikTok and carried out by US market monitor Luminate, 62% of US TikTok users pay for a music streaming service,
compared to 43% of all consumers.
</item 7>

<item 8>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“This preference, combined with the high volume of music discovery and sharing activity on TikTok, can be harnessed to grow a fanbase without borders and has
helped to launch global artists from genres such as Reggaeton, Afrobeats and K-Pop.”

The report added: “TikTok users‘ interest in international music is evident in their music streaming preferences. Compared to the average music listener across
each market, this audience is more likely to indicate that ‘access to music by global artists’ is an extremely important factor when selecting a music streaming
service.”

“TikTok is already famous for being the launchpad for creating viral hits and breaking new artists, but we wanted to understand how TikTok and its users impact
the music industry more broadly,” Ole Obermann, TikTok’s Global Head of Music, said in a statement.
</item 8>

<item 9>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Third-party Reddit app Narwhal says it hopes to survive Reddit’s app purge with a subscription plan.

Apropos “getting together,” Amanda reports that VidCon is still kickin’. For the first time, the conference hosted an industry leadership summit, where creator
economy experts and creators could hash out their grievances with the state of the business and share ideas to make the job of a creator more sustainable. That
makes sense, unlike creators raising venture capital: It’s so eye-wateringly hard to make money as a creator, I’d love to see the pitch that convinces a VC to
cut a check to a creator, with a realistic expectation of a venture-scale return.

Maybe they were hiding behind the sofa?: Sarah reports that Mastodon actually has 407K+ more monthly users than it thought — and it seems like Twitter has a lot
more traffic than Musk said. Still, the peak now is about the same as it was a decade ago, and it’s unclear what the social platform can do to encourage more
growth.
</item 9>

<item 10>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“Luminate’s findings prove what we had long known: that TikTok is the driving force behind music discovery in the industry, and that TikTok users are active,
engaged and highly valuable drivers of music industry revenues.”

The new study is part of an ongoing effort by TikTok to highlight its growing role in the development of global music culture. Last year, TikTok released a
report asserting that 13 out of the 14 songs to reach number on the US Billboard Hot 100 in 2022 were driven by viral trends on TikTok.

The data clearly shows why TikTok has focused so heavily on developing music-related services, to the point that one could be forgiven for thinking the company
plans to reinvent itself as a part of the music industry.

Last year, TikTok launched SoundOn, its own music distribution service, which has partnered with Believe-owned distro company TuneCore and Downtown
Holdings-owned B2B tech and services company FUGA.
</item 10>
=== END ITEM id=d4661944c14f4271 ===

=== ITEM id=55d05b4312b8e59b ===
Gold sentences and facts:
Sentence 0: "We all found ourselves at home looking at the people we worked with, the people we went out for drinks with and the people in our extended family on a computer screen," he told Fox News Digital.
  Fact 0.0: A speaker said that they all found themselves at home looking at the people they worked with, the people they went out for drinks with, and the people in their extended family on a computer screen.
  Fact 0.1: The speaker told Fox News Digital this.
Sentence 1: We used to play in person, but since the pandemic I run the game virtually online from my house.
  Fact 1.0: The speaker used to play the game in person.
  Fact 1.1: Since the pandemic, the speaker runs the game virtually online from their house.

Retrieved items:
<item 1>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
“It was a huge demand to come back. I think fandoms have grown in that time period, just the fervor around specific creators that was totally and wholly
different than we’d seen before,” Delphin said of last year’s TwitchCon. “And the creators themselves had also absolutely exploded. It used to be that you’d
come to TwitchCon, be the biggest creator, and you could walk around pretty freely. Now it’s huge, it’s a whole different kind of apparatus to manage.”

Attendee feedback is clear: Next year’s convention, wherever it is, needs to have more accessible third places.

The third place is a sociological term that refers to a physical space that isn’t work or home, where people can socialize without the obligations of
productivity. If you can hang out there, it’s a third place. During the pandemic, virtual communities served as ad hoc third places in lieu of coffee shops,
bookstores and gyms. Twitch, which saw exponential growth during the pandemic, was one of them.
</item 1>

<item 2>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Our campaign is called The Heartsguard Saga, and it’s based off Deven Rue’s map of a place called Euphoros. My players are at around level 15 and I frequently
give them a variety of amazing and unique gear that I’ve dreamed up all on my own... which they go on to forget they have, and never use. We used to play in
person, but since the pandemic I run the game virtually online from my house. I put a camera hovering over my gaming mat, and make a labeled grid for the
players so they call out, “Move me from A5 to B10.” During important rolls, I even make my players face their camera toward their dice for added suspense!

Petrana Radulovic, entertainment reporter

I’d wanted to play Dungeons & Dragons since I was in high school, but I never knew how to get into it.
</item 2>

<item 3>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Thankfully, in the doldrums of 2020, someone I knew knew someone who knew someone (that’s literally what the connection was) who was interested in running a
game. That kicked off my D&D career, but was this year that everything finally clicked in my brain and it went from a hobby I enjoyed to one that I spend every
free minute of my spare time thinking about.

Part of it is finally having reliable groups. After some initial shuffling of members, in 2023 I’ve been blessed to have not one but two separate D&D groups,
both with enthusiastic members who are committed to at least trying to meet regularly. And yes, part of it is Baldur’s Gate 3, which helped me to understand
some of the more complicated D&D mechanics and just generally made me very excited about combat in a way I never was before.

I’ve still yet to actually have a game in person! My groups are spread out across North America, but I’m hoping to manifest an in-person session soon.
</item 3>

<item 4>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
Brands, too: Remember the Tony the Tiger VTuber in August of 2022?

Many creators that jumped on the trend have since stopped using their avatars, but at the start of the COVID-19 pandemic, there was a large increase in VTubers
on Twitch and other streaming platforms, like YouTube, as models and equipment became more accessible to the general public. Twitch has since gone on to
highlight a wide variety of VTubers through its VTuber Takeover promotion, which most recently ran in August. Twitch also reported that VTubing-related content
on the platform shot up by some 350% between January and August 2022. Despite this, some have argued that the VTubing bubble has all but burst.

Polygon is celebrating TwitchCon 2023 with a package of stories covering streaming trends and news from the show. Check it out!

“I do feel that VTubing has become less novel as the market has become more saturated,” says a streamer with agency Nijisanji who is not approved to speak
publicly.
</item 4>

<item 5>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CAN YOU AVOID END-OF-THE-YEAR CAREER BURNOUT? JOB EXPERTS REVEAL SECRETS

Corcoran agreed, noting that many report feeling livelier, happier and more active during in-person meetings.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions," she told Fox News Digital.

"As social beings, we’re highly attuned to these cues, and it takes more work for us to look for and process them in the virtual environment, leading to the
fatigue we experience."

Even beyond the workplace, Elliott noted that people are using video in more personal contexts than ever before, such as sharing reels on social media and
calling loved ones on FaceTime.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions."

"This can further exacerbate the problem if you're already spending a significant portion of your day on Zoom," he said.
</item 5>

<item 6>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.
</item 6>

<item 7>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
After more than a decade of going on and on about the “board gaming renaissance,” it’s safe to say that tabletop role-play is finally having its moment in the
spotlight. But while Dungeons & Dragons’ OGL fiasco has clearly lit a fire under many players eager to make a change, the fact of the matter is that a sizeable
cohort of writers and designers have been toiling away at excellent games for years now. Their tireless efforts have yielded a bumper crop of excellent, some
might say genre-defining, TTRPGs. Just like board games, a few modern classics have also cropped up: Look no further than The Quiet Year and The One Ring, both
making their second appearance on this list.

Just as before, Polygon asked nearly two dozen writers, designers, presenters, actors, and personalities from around the world of tabletop gaming to share with
our readers the TTRPGs that made an impact on them and their players this year. Here’s what we found.
</item 7>

<item 8>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
While the playing habits of a Twitch streamer don’t represent the average player by any means, content creators certainly shape public opinion about games.

“Negative sentiment is very profitable,” says the AAA game developer, who noted that social media platforms are built to highlight popular — and sometimes
caustic — content. This dynamic sets up an inherent tension between game developers and hardcore players like Twitch streamers, who are not the main audience of
most games but are often some of the most visible players.

“I don’t at all think that streamers are embodying some typical player behaviors, like ‘Brandon,’” the developer continues, posing a name for an imaginary
everyday consumer. “There’s definitely a model of consumers nowadays who probably parrot those sentiments. They did not arrive at those conclusions themselves.”

“There is inherently [a] vicious and negative cycle for everyone involved, for how live service interacts with Twitch as a medium”

The misinformation will vary, but Twitch both shapes that discourse while also becoming the very metric that people use to make sense of a game’s situation.
</item 8>

<item 9>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
Julia Corcoran, a licensed clinical psychologist and board-certified coach, who is currently serving as the director of clinical strategy and experience at
Modern Health in San Francisco.

The challenge isn’t exclusive to Zoom; other videoconferencing platforms can have the same effect.

The pandemic's effect

With the COVID-19 pandemic, Zoom and other videoconferencing platforms "exploded in popularity," said Dr. Carl Nassar, PhD, a professional counselor based in
Denver, Colorado.

"We all found ourselves at home looking at the people we worked with, the people we went out for drinks with and the people in our extended family on a computer
screen," he told Fox News Digital.

WORKERS FIRED ON ZOOM: HERE'S HOW TO NAVIGATE A 'DEVASTATING' EXPERIENCE AND COME OUT ON THE OTHER SIDE

"Our work went virtual, our social life went virtual, our doctors' appointments went virtual, and even our intimate relationships sometimes went virtual," he
said.
</item 9>

<item 10>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 10>
=== END ITEM id=55d05b4312b8e59b ===
