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

=== ITEM id=6056ae189b84927b ===
Gold sentences and facts:
Sentence 0: It was a Michigan quarterback that led the Buccaneers to the promised land last time so perhaps lightning strikes twice.
  Fact 0.0: A Michigan quarterback led the Buccaneers to the promised land last time.
  Fact 0.1: The text suggests that lightning might strike twice.
Sentence 1: Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh.
  Fact 1.0: The Big Ten is reviewing the lengthy written responses from Michigan.
  Fact 1.1: The Big Ten is reviewing the lengthy written responses from Jim Harbaugh.
  Fact 1.2: The source said this.

Retrieved items:
<item 1>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will Harbaugh finally take the parachute to the NFL? Those are questions bound to come up at the end of the 2023 season.

BENDER: Michigan will need Jim Harbaugh against Ohio State

Big Ten

This is the fourth football scandal in less than a calendar year.

Three former Iowa football players and a student manager were charged in an illegal sports wagering scandal. Several student-athletes face allegations of
placing bets on games in which they played in. Northwestern coach Pat Fitzgerald was fired on July 10 after a hazing investigation into the program. Fitzgerald
is suing the university for wrongful termination. Michigan State fired Mel Tucker on Sept. 27 after allegations of sexual misconduct from Brenda Tracy, a
speaker on sexual assault awareness.

Now, Michigan has a scandal on the field, and they countered with allegations that Ohio State and Rutgers leaked Michigan's signs to Purdue in last year's Big
Ten championship game.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Michigan, Michigan sign-stealing scandal, Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti, Pat Fitzgerald,
Northwestern, Brenda Tracy, Mel Tucker, Rutgers, illegal sports wagering scandal, hazing investigation, wrongful termination, sexual assault awareness, leaking
Michigan's signs, Big Ten
</item 1>

<item 2>
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

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Michigan, Michigan sign-stealing scandal, CBSSports.com, NCAA rules committee, coach-to-player helmet communication, bowl season, Nittany Lions,
Saturday’s victory against the Nittany Lions, Beaver Stadium, Sherrone Moore, Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony
Petitti, Jim Harbaugh, Sporting News, Tony Petitti
</item 2>

<item 3>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
2 now, and the debate with the Buckeyes and No. 1 Georgia is going to perk up more than ever. They are the three best teams in the country, but only one has
uncertainty with the head coach.

Whether they admit it or not, Michigan still needs Harbaugh for those moments, especially in two weeks.

The reality is he might not be present for that one either.

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan

Entities: Michigan, Penn State, Jim Harbaugh, Ohio State, Sporting News, Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different
game, Georgia
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
However, the primary difference is Michigan is accused of using in-person scouting, which has been prohibited under NCAA rules for nearly 30 years, to acquire
its information.

Monday, Nov. 6

No connection between Buckeyes coach, Michigan investigation

Ohio State coach Ryan Day and his family have no connection to the Big Ten and NCAA investigations into Michigan, Yahoo Sports reported.

A report emerged over the weekend indicating Day may have had some involvement in the probe into the Buckeyes' arch-rivals, but the NCAA told the Big Ten there
are no ties that it knows of between the Ohio State coach and the investigation.

NCAA shares findings with Big Ten as Harbaugh suspension looms

Yahoo Sports reported Monday that the NCAA shared its investigation findings with the Big Ten, noting that the findings "do not connect the in-person scouting
and recording of opponents’ sidelines to Harbaugh."

Still, the conference is considering a suspension for Harbaugh and could hand down punishments before the end of the week.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, in-person scouting, Big Ten, Yahoo Sports, Jim Harbaugh, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh
suspension from NCAA cheating investigation, Big Ten investigation into Michigan, NCAA investigation into Michigan, possible suspension for Jim Harbaugh,
recording of opponents’ sidelines, NCAA rules, Ryan Day, Ohio State, Sporting News, NCAA
</item 4>

<item 5>
Graph query row:
article: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
publisher: Sporting News
mentioned_entities: ['NFL', 'The Athletic', 'Michigan', 'Connor Stallions', 'in-person scouting', 'sign stealing', 'scandal', 'showdown against Penn State',
'Nicole Auerbach', 'sign-stealing scandal', 'Big Two', 'DeCourcy', 'Sports Media Watch', 'Michigan sign-stealing scandal', 'Harbaugh suspension', "Friday's
announcement", 'sportsmanship', 'television ratings', 'college football', 'wrongdoing', 'legal response', 'Penn State', 'Day', 'College Football Playoff',
'national championship', 'WWE', 'CBSSports.com', 'NCAA rules committee', 'coach-to-player helmet communication', 'bowl season', 'Nittany Lions', 'Saturday’s
victory against the Nittany Lions', 'Beaver Stadium', 'Sherrone Moore', 'defeating Penn State', 'NCAA investigation into whether Michigan was involved with
in-person scouting', 'CFP', 'Harvey Updyke', 'Nevin Shapiro', 'ESPN.com', "Chippewas' opener against Michigan State", 'Stalion', 'Michigan fans', 'Maryland',
'temporary restraining order hearing', 'rule-breaking', '2020 season', 'Pat Fitzgerald', 'Northwestern', 'Brenda Tracy', 'Mel Tucker', 'Rutgers', 'illegal
sports wagering scandal', 'hazing investigation', 'wrongful termination', 'sexual assault awareness', "leaking Michigan's signs", 'Greg Sankey', 'Big Ten',
'in-person scouting scandal', 'Santa Ono', 'J.J. McCarthy', 'Blake Corum', 'Ric Flair', 'Terps', 'Buckeyes', 'ESPN', 'USC', 'MLB Network', 'Stalions', 'Jim
Harbaugh', 'Warde Manuel', 'game against Penn State', 'Connor Stalions', 'the game', 'Big Ten championship game', 'Ohio State', 'UCLA', 'Colorado', 'Iowa',
'Washington', 'Michigan-Penn State game', 'SEC', 'Notre Dame', 'Oregon', 'Michigan State', 'Big Ten East', 'Nielsen', 'Tony Petitti', 'sign-stealing', 'NCAA
investigation', 'Central Michigan', 'NCAA', 'Purdue', 'sexual misconduct']
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Although the Big Ten has closed its investigation, we are continuing to cooperate with the NCAA as it moves forward with its ongoing investigation. Consistent
with our commitment to integrity, we will continue to take the appropriate actions, including disciplinary measures, based on information we obtain. Earlier
today, Michigan Athletics relieved Chris Partridge of his duties as a member of the Michigan Football staff. Due to employee privacy laws, we are unable to
comment further."

Because Michigan and Harbaugh on Thursday dropped their litigation against the Big Ten, he will not be on Michigan's sideline this week against Maryland and
next week against Ohio State to end the regular season.

Former Wolverines assistant and Harbaugh confidant Biff Poggi on Friday offered his former boss an invitation. Poggi, in his first year as Charlotte's head
coach, asked Harbaugh to join the 49ers for their games vs. Rice on Saturday and at South Florida on Nov. 25.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Maryland, Big Ten, Chris Partridge, Michigan Football, Biff Poggi, Charlotte, South Florida, Jim Harbaugh, NCAA
cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Michigan
Athletics, Ohio State, Sporting News
</item 6>

<item 7>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Michigan sign-stealing scandal, Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti, Santa Ono, J.J.
McCarthy, Blake Corum, Ric Flair, Terps, Buckeyes, Jim Harbaugh, Warde Manuel, Sporting News, Tony Petitti
</item 7>

<item 8>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan

Entities: Michigan, Penn State, temporary restraining order hearing, Big Ten, Blake Corum, ESPN, Jim Harbaugh, Ohio State, Conor Stallions, College Football
Playoff, Sporting News, in-person scouting and sign-stealing scandal, Abdul Carter, Michigan-Penn State game, Michigan didn't need Jim Harbaugh against Penn
State, but Ohio State is a different game
</item 8>

<item 9>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Michigan, in-person scouting, Stalion, Michigan fans, ESPN.com, Maryland, temporary restraining order hearing, rule-breaking, 2020 season,
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti, Big Ten, Jim Harbaugh, Ohio State, Sporting News
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, Michigan, Rutgers, Big Ten, Jim Harbaugh, NCAA cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible
Jim Harbaugh suspension from NCAA cheating investigation, Michigan-Purdue Big Ten Championship Game, decoded signals, Pete Thamel, Ohio State, Sporting News,
NCAA, Purdue
</item 10>
=== END ITEM id=6056ae189b84927b ===

=== ITEM id=130ab1d7990b02d4 ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: That allows sportsbooks to earn the profit built into the odds and lines without worrying about which side wins.
  Fact 1.0: This allows sportsbooks to earn the profit built into the odds and lines.
  Fact 1.1: Sportsbooks earn that profit without worrying about which side wins.

Retrieved items:
<item 1>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Rookie of the Year is not the only form of NBA futures betting. Legal online sportsbooks have a wide selection of NBA futures you can bet on, including:

In addition to futures bets, you can wager on many NBA bets during the regular season and postseason. NBA game lines (moneylines, point spreads, and totals) are
the most popular, but you’ll also be able to bet on NBA team and player props, parlays, and more.

The NBA is the second-most popular odds market to bet on in the US, behind NFL betting odds. That means every online betting site always has a great selection
of NBA odds.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Wembanyama, NBA Rookie of the Year Odds - Wembanyama the early favorite, US, Sporting News, Rookie of the Year, NBA
</item 1>

<item 2>
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

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA betting, basketball odds, spreads, totals, parlays, online sportsbooks, The Best NBA Betting Sites and Apps for the 2023-24 Season,
NBA, NBA prop bets, moneylines
</item 2>

<item 3>
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

Graph facts:
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> live odds feed
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> NFL
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> odds and lines
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> How to bet on sports in the US
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> sportsbook apps

Entities: line shopping, Line Shopping in Sports Betting - How to Find the Best Odds, parlay bet, roster move, Sporting News, Betting patterns, injury, sports
betting, Odds
</item 3>

<item 4>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
We urge you to read the requirements for any available promotions and offers before accepting them so you’re aware of any restrictions before betting.

You can now place your initial deposit, which is easy at online betting sites thanks to various deposit and withdrawal options.

To bet on the NBA Rookie of the Year, navigate to the basketball section of the sportsbook website, select the NBA, and go to the futures tab.

You’ll find Rookie of the Year odds, NBA MVP odds, and more there. Select the bet you want to make, enter the amount you want to bet, and finalize your bet
slip.

Ensure you’re using only legal online sportsbooks for any type of betting. Offshore sportsbooks advertise widely on the internet, but they are illegal and offer
no protection against fraud and other criminal activities.

Best sportsbooks for NBA Rookie of the Year odds

Here are some of the best sports betting sites where you can bet on the next NBA Rookie of the Year.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA Rookie of the Year award

Entities: Wembanyama, NBA Rookie of the Year, NBA Rookie of the Year Odds - Wembanyama the early favorite, Sporting News, NBA MVP, NBA
</item 4>

<item 5>
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
</item 5>

<item 6>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.

Graph facts:
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> live odds feed
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> NFL
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> odds and lines
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> How to bet on sports in the US
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> sportsbook apps

Entities: line shopping, Line Shopping in Sports Betting - How to Find the Best Odds, Sporting News, sportsbooks, sports betting
</item 6>

<item 7>
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

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, live betting, DraftKings, The Best NBA Betting Sites and Apps for the 2023-24 Season, BetMGM, FanDuel, NBA
</item 7>

<item 8>
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

Graph facts:
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> NFL
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> line shopping
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> odds and lines
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> How to bet on sports in the US
- Line Shopping in Sports Betting - How to Find the Best Odds -MENTIONS-> live odds feed

Entities: Line Shopping in Sports Betting - How to Find the Best Odds, Vegas line, Las Vegas sportsbooks, online betting sites, Sporting News, sports betting,
Las Vegas
</item 8>

<item 9>
Graph query row:
article: Line Shopping in Sports Betting - How to Find the Best Odds
publishers: ['Sporting News']
passages: [{'passage': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nLine shopping is the key that
unlocks the best odds and lines at top sports betting sites.\n\nThe process is as simple as checking the lines and odds for your prospective bet at multiple
online sports betting sites. The result is quite often better odds and improved lines.\n\nWhat is line shopping, and why is it worth your time? Find the answers
below in our complete guide to line shopping.\n\nHow do sportsbooks set odds and lines?\n\nThe best sportsbooks employ oddsmakers and oddsmaking services to set
their lines and odds.\n\nOther sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that
consider analytics, statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.\n\nBetting lines and odds are then
subject to change based on the betting public’s reaction and any additional information that comes to light ahead of the game. Sportsbooks will move the line or
change the odds if the public bets heavily on one side.', 'page': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News,
2023-11-06T22:21:05+00:00)'}, {'passage': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nSportsbooks
will move the line or change the odds if the public bets heavily on one side.\n\nEvery sportsbook’s goal is to even out the betting on both sides of every bet.
That allows sportsbooks to earn the profit built into the odds and lines without worrying about which side wins.\n\nThe betting is rarely the same at every
sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t. That’s when you’ll spot a difference between
sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.\n\nThe lines and odds may even be different right from the
jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can find opportunities to lay less, win more, and improve your
chances.', 'page': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)'}, {'passage': 'Line Shopping in
Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nShop around, and you can find opportunities to lay less, win more, and
improve your chances.\n\nSee more: How to bet on sports in the US\n\nShop for the best lines at online sportsbooks\n\nBelow is a live odds feed that shows you
the differences between lines and odds for the same bets at many of the top sportsbooks in the country. Look here for current opportunities.\n\nWhat’s the best
way to shop lines?\n\nThe easiest way to line-shop is to find something like the live odds feed above. It’s a one-stop shop. Shopping lines here will take just
seconds.\n\nPerforming the task manually may take longer, but you can check with more sportsbooks or the chosen few you feel consistently offer the best lines
and odds. The best way to do this is to download and install more than one of the best sports betting apps on your phone.\n\nThen, tap around among the
sportsbooks to compare lines and odds for the bets you plan.\n\nBest sportsbooks for line shopping\n\nThe biggest brands in sports betting offer competitive
lines in dozens of states.', 'page': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)'}, {'passage':
"Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nShop around at the following top online sportsbooks to
find the best prices on your bets.\n\nCaesars Sportsbook\n\nCaesars' odds and lines are consistently among the best in the country, making it a must-see when
shopping lines. The point spreads here can be a half-point to two points better than the competition’s, and the moneyline odds may be more favorable.\n\nBetMGM
Sportsbook\n\nThe lines and odds at BetMGM are consistently competitive. Plus, sports betting promos and odds boosts abound, so you'll find a better price on
many bets here.\n\nDraftKings Sportsbook\n\nDraftKings has more open betting markets than most online sportsbooks, and the lines and odds are always
competitive.\n\nFanDuel Sportsbook\n\nFanDuel is committed to posting competitive odds and runs so many generous promos and odds boosts that it is always worth
a look.\n\nHow to find the best lines and odds for any bet\n\nShopping around lets you find the best lines and odds for any bet.", 'page': 'Line Shopping in
Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)'}, {'passage': 'Line Shopping in Sports Betting - How to Find the Best
Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nThat means checking the posted lines and odds for the same bets at different sportsbooks.\n\nYou can do this
for moneylines, point spreads, totals bets, prop bets, and more. Finding the best line or better odds on any bet is like finding money.\n\nYou can shop around
using multiple sportsbook apps on your phone or live odds feeds like the one above. Pick a side, then use the drop-down menus and tabs to find the odds and
lines for that bet.\n\nFinally, bet when you see an opportunity at better odds or an improved line compared to the competition.\n\nLine shopping for
futures\n\nPeople tend to bet futures at their favorite online sportsbook. However, shopping for better odds on any future bet is worth your time. Most game
lines are similar at different sportsbooks, but the odds in futures markets can vary significantly.\n\nSportsbooks sometimes have higher liabilities for
specific teams and offer better odds on others to lay that off. Timing is everything.\n\nGet it right, and you can easily take advantage of changes in different
futures markets due to liability.', 'page': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)'},
{'passage': "Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nGetting more points or laying fewer on a
typical NFL point spread or NBA point spread might lead to winning a bet you’d otherwise lose. Plus, if you find better odds on any bet, you can potentially
drop fewer dollars when you lose and earn more when you win.\n\nShopping lines will help you consistently find the best prices on your bets. It won't help you
pick more winners, but it will improve your chances and lead to bigger wins and more affordable losses. That’s undoubtedly worth your time.\n\nFor example, you
might like the Kansas City Chiefs to beat the Cincinnati Bengals in a mid-season NFL game. Say you look around and see that three of four licensed online
sportsbooks in your state have the Chiefs as -3 favorites, but the fourth has them at just -2.5.\n\nLet’s say the Chiefs win by exactly three, a distinct
possibility since a single field goal decides most NFL games. Bets at the first three sportsbooks would push. However, you’d have a winner at the fourth.",
'page': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)'}, {'passage': 'Line Shopping in Sports Betting
- How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nHowever, you’d have a winner at the fourth. If you didn’t shop around, you’d never have
found that half-point victory and be stuck with a push instead of a profit.\n\nWhat is the Vegas line?\n\nLas Vegas sportsbooks set the standard for lines and
odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line. Most online betting sites don’t stray too far from the Vegas
line, using it as a foundation for setting all lines in every major sport.\n\nIf those lines do shift, it’s usually because of increased betting on one side of
a bet. Shop lines locally, and you can take advantage when sportsbooks stray from the Vegas line.\n\nWhy do the lines move?\n\nIt’s worth reiterating that
sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they earn the small profit built into the odds without concern about
who wins or loses.', 'page': 'Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)'}, {'passage': 'Line
Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)\nSportsbooks want even betting so much that they’ll move a
line or change the odds to draw more bets to the other side to get it that way.\n\nThat’s when you’ll see a difference between the lines and odds. Every
sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move. It’ll be difficult for you to find value in
that case.\n\nBut when a line moves or odds change because of betting patterns, only the sportsbook that saw the betting makes the change. That’s your
opportunity right there.\n\nLine shopping is a tool to help you spot changes due to betting patterns. Then, you can simply take advantage of movement in a line
or changes in the odds.\n\nChoose a side, shop around at online sportsbooks, compare the lines and odds, and then pounce when you find one that has something
different and more favorable to your viewpoint.\n\nSee more: What is a parlay bet?', 'page': 'Line Shopping in Sports Betting - How to Find the Best Odds
(Sporting News, 2023-11-06T22:21:05+00:00)'}]
connections: ['NFL', 'line shopping', 'odds and lines', 'How to bet on sports in the US', 'live odds feed', 'sportsbook apps', 'live odds feeds', 'totals bets',
'futures markets', 'liability', 'teams', 'Vegas line', 'Las Vegas sportsbooks', 'online betting sites', 'parlay bet', 'roster move', 'Cincinnati Bengals', 'US',
'point spread', 'sportsbooks', 'sports betting apps', 'Betting patterns', 'DraftKings Sportsbook', 'FanDuel Sportsbook', 'BetMGM Sportsbook', 'injury', 'Caesars
Sportsbook', 'NBA', 'prop bets', 'sports betting', 'Kansas City Chiefs', 'Las Vegas', 'point spreads', 'moneylines', 'Odds']
</item 9>

<item 10>
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

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA betting, NBA, sportsbooks, responsible gaming, The Best NBA Betting Sites and Apps for the 2023-24 Season, cybersecurity
</item 10>
=== END ITEM id=130ab1d7990b02d4 ===

=== ITEM id=5741eb57fc6fd6cd ===
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

Graph facts:
- Top 25 MLB offseason trade candidates -MENTIONS-> Blue Jays
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Blue Jays
- Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant -MENTIONS-> Blue Jays
- World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 -MENTIONS-> Blue Jays
- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings -MENTIONS-> Blue Jays

Entities: Blue Jays, Soto, Randy Arozarena, Rays, Mariners, Ohtani, 2023 free agency, MLB Winter Meetings tracker: Live updates on news, rumors, signings and
trades for 2023 free agency, MLB Network, Jon Morosi, Winter meetings, Isaac Paredes, Vladimir Guerrero Jr., Bo Bichette, George Springer
</item 1>

<item 2>
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

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> MLB
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> MLB
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> MLB
- Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00) -MENTIONS-> MLB
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> MLB

Entities: MLB, Padres, Tigers, Red Sox, Rockies, Nationals, Guardians, Cal Quantrill, Twins, Kyle Farmer, Rays, Harold Ramírez, Manuel Margot, Mariners, Top 25
MLB offseason trade candidates
</item 2>

<item 3>
Graph query row:
requested_entity: jerry dipoto
entity: Jerry Dipoto
publisher: Yardbarker
page: Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
passage: Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
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
</item 3>

<item 4>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
New York has subsequently walked back some of their statements about a possible retool. They’re still aiming to compete in ’24 even if they’re signaling they
won’t blow away the top of the free agent market.

First-year president of baseball operations David Stearns has twice publicly said he anticipates Alonso being the Opening Day first baseman in Queens. That
won’t stop teams from trying, but an offseason deal looks decidedly unlikely.

Others of note

A’s: Seth Brown

Braves: Vaughn Grissom

Brewers: Adrian Houser, Devin Williams, Brandon Woodruff

Cardinals: Alec Burleson, Tommy Edman, Nolan Gorman, Tyler O’Neill

Giants: J.D.

Graph facts:
- Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? -MENTIONS-> David Stearns
- Dennis Eckersley -WORKS_FOR-> A’s
- Paul Blackburn -WORKS_FOR-> A’s
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> A’s
- David Stearns -WORKS_FOR-> Mets

Entities: A’s, New York, David Stearns, Alonso, Opening Day, Seth Brown, Vaughn Grissom, Brewers, Adrian Houser, Devin Williams, Brandon Woodruff, Cardinals,
Alec Burleson, Tommy Edman, Nolan Gorman
</item 4>

<item 5>
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

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Minnesota
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Minnesota
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -MENTIONS-> Minnesota
- Top 25 MLB offseason trade candidates -MENTIONS-> Minnesota
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Minnesota

Entities: Minnesota, Yankees, Max Kepler, Jorge Polanco, Twins, Edouard Julien, Royce Lewis, 2023 Winter Meetings, MLB Network, Jon Morosi, Sporting News,
Yamamoto, AL Central title, MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings
</item 5>

<item 6>
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

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> MLB
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> MLB
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> MLB
- Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00) -MENTIONS-> MLB
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> MLB

Entities: MLB, Minnesota Twins, Tigers, Mark Canha, GM Meetings, A's, Rockies, Nationals, 2024 season, payroll, local television rights contracts, offseason
trades, Milwaukee Brewers, White Sox, Angels
</item 7>

<item 8>
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

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> MLB
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> MLB
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> MLB
- Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00) -MENTIONS-> MLB
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> MLB

Entities: MLB, Burnes, Adames, Guardians, Corey Kluber, Mike Clevinger, Carlos Carrasco, Top 25 MLB offseason trade candidates, Shane Bieber, Yardbarker
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

Graph facts:
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> MLB
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> MLB
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> MLB
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> MLB
- Top 25 MLB offseason trade candidates -MENTIONS-> MLB

Entities: MLB, Bryan Woo, Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00), Seattle Mariners, Bryce Miller, Emerson Hancock, George
Kirby, Logan Gilbert, Jerry Dipoto, Yardbarker
</item 9>

<item 10>
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

Graph facts:
- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings -MENTIONS-> Dylan Cease
- Carlos Correa -WORKS_FOR-> Twins
- Top 25 MLB offseason trade candidates -MENTIONS-> Dylan Cease
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Carlos Correa
- Top 25 MLB offseason trade candidates -MENTIONS-> White Sox

Entities: Carlos Correa, Dylan Cease, White Sox, Rays, Mariners, Ohtani, 2023 free agency, Orioles, MLB Winter Meetings tracker: Live updates on news, rumors,
signings and trades for 2023 free agency, Yoshinobu Yamamoto, Braves, Jon Morosi, MLB Network, Susan Slusser, San Francisco Chronicle
</item 10>
=== END ITEM id=5741eb57fc6fd6cd ===
