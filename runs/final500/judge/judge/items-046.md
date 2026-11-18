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

=== ITEM id=4cf30d3baaef01b3 ===
Gold sentences and facts:
Sentence 0: James is seeking $250 million in penalties and a ban on Trump doing business in New York, his home state.
  Fact 0.0: James is seeking $250 million in penalties.
  Fact 0.1: James is seeking a ban on Trump doing business in New York.
  Fact 0.2: New York is Trump's home state.
Sentence 1: No apartment in New York City has ever sold for close to that amount, James said.
  Fact 1.0: No apartment in New York City has ever sold for close to that amount, according to James.
Sentence 2: The prosecution argues that was to mask a drop in the value of one of his other properties.
  Fact 2.0: The prosecution argues that a transaction was meant to mask a drop in the value of one of Trump's other properties.

Retrieved items:
<item 1>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
James, a Democrat, sued Trump and the Trump Organization a year ago, alleging a pattern of duplicity that she dubbed “the art of the steal,” a twist on the
title of Trump’s 1987 business memoir “The Art of the Deal.”

The lawsuit accused Trump and his company of routinely inflating the value of assets like skyscrapers, golf courses and his Mar-a-Lago estate in Florida,
padding his bottom line by billions.

Among the allegations were that Trump claimed his Trump Tower apartment in Manhattan — a three-story penthouse replete with gold-plated fixtures — was nearly
three times its actual size and valued the property at $327 million. No apartment in New York City has ever sold for close to that amount, James said.

Trump valued Mar-a-Lago as high as $739 million — more than 10 times a more reasonable estimate of its worth.
</item 1>

<item 2>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
Consequently, Palm Beach values the property as a business on its tax rolls. In 2021, the county assessed it at $US27.6 million, lower than comparable
residential properties. New York State argues that this appraisal is the one Trump should have been using. New York’s estimated value “was a shock to the real
estate community and anybody with any understanding of the island and its values,” said Liza Pulitzer, a real estate agent at Brown Harris Stevens and a Palm
Beach native. “It could easily fetch $US500 million or more.”

That assumes a buyer views it as a single-family residence. It’s currently not zoned for such use, though Trump is able to reside there through a loophole
designating himself an employee. Trump’s legal team engaged an expert witness who argued that a future buyer could do the same thing. And if they didn’t like
the idea of sharing their property with others, they could reduce the club to a membership of one.
</item 2>

<item 3>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
40 Wall Street Trump’s valuation (2021): $US664 million, Bloomberg valuation (2023): $US270 million, New York’s valuation: No current estimate available In
valuing his flagship lower Manhattan office tower, Trump consistently ignored more conservative appraisals he’d received from professionals, New York State
argues. Trump has been accused of significantly over-valuing his flagship downtown New York property. Credit: Bloomberg For example, 40 Wall Street was
appraised by a real estate company at $US200 million in 2011 and $US220 million in 2012. Despite this, Trump valued the property in financial statements at
$US524.7 million and $US527.2 million, respectively, for those years.

Bloomberg uses a capitalisation of net income approach in coming up with an estimate. Valuations for commercial offices across the board have taken a hit in the
past 18 months amid rising interest rates and a lacklustre corporate leasing market. But 40 Wall Street has its own unique problems.
</item 3>

<item 4>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
The following year, Trump valued them at $US4.1 million each — more than 65 times greater. Trump has continued to value stabilised apartments in the building at
market rates, including in his 2021 statement of financial condition. Rent-stabilised units “with a positive cash flow are often valued at 15-25 cents a dollar
of the open market value, versus single digits for negative cash flow,” said Jonathan Miller, president of appraiser Miller Samuel. Still, Trump’s optimism was
shown to be at least partially justified. By 2021, only five of the 12 apartments remained rent stabilised. The prosecution also claims two additional
non-stabilised apartments in the building were overvalued because Trump ignored an option his daughter, Ivanka Trump, had to buy them at a price lower than the
market value he used.
</item 4>

<item 5>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
Trump’s Fifth Avenue penthouse Trump’s valuation (2021): $US131.3 million, Bloomberg valuation (2023): $US40 million, New York valuation: No current estimate
available Trump’s penthouse apartment at Trump Tower has been an integral part of his image for 40 years, so it’s not surprising that one of the world’s masters
of hyperbole exaggerated its size for most of that time. Trump’s penthouse apartment at Trump Tower has been an integral part of his image for 40 years. Credit:
AP Between 2011 and 2016, Trump valued his apartment based on it being 30,000 square feet (2,787 square metres) in size — almost triple its actual measurement
of 10,996 square feet. Because of this, New York State arrived at a valuation that was about a third of Trump’s in those years. They haven’t provided a more
recent estimate.

Space inflation wasn’t the only issue.
</item 5>

<item 6>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
But it’s also likely that the association with Trump would turn off some buyers. Trump Park Avenue Trump’s valuation (2021): $US90.9 million, Bloomberg
valuation (2023): $US86.4 million, New York’s valuation (2021): $US80 million The residential condo tower on Manhattan’s Upper East Side, formerly the Hotel
Delmonico, isn’t one of Trump’s best-known buildings, but it’s been the source of some significant asset inflations, according to the prosecution.

Its value to Trump mostly stems from unsold condo units that are owned by the Trump Organisation and rented out. A varying number have been rent-stabilised,
meaning tenants pay below market rates. Typically, these apartments are valued at a discount because tenants can stay indefinitely, sometimes even passing them
on to children. In 2010, 12 rent-stabilised apartments owned by Trump were valued at an average of $US62,500 each in an appraisal provided by a real estate
company.
</item 6>

<item 7>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
The company was fined $1.6 million. One of the executives, Trump’s longtime finance chief Allen Weisselberg, pleaded guilty and served five months in jail. He
is a defendant in James’ lawsuit and gave sworn deposition testimony for the case in May.

James’ lawsuit does not carry the potential of prison time, but could complicate his ability to transact real estate deals. It could also stain his legacy as a
developer.

James has asked Engoron to ban Trump and his three eldest children from ever again running a company based New York. She also wants Trump and the Trump
Organization barred from entering into commercial real estate acquisitions for five years, among other sanctions. The $250 million in penalties she is seeking
is the estimated worth of benefits derived from the alleged fraud, she said.
</item 7>

<item 8>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
Beyond mere bragging about his riches, Trump, his company and key executives repeatedly lied about them on his annual financial statements, reaping rewards such
as favorable loan terms and lower insurance premiums, Engoron found.

Those tactics crossed a line and violated the law, the judge said, rejecting Trump’s contention that a disclaimer on the financial statements absolved him of
any wrongdoing.

“In defendants’ world: rent regulated apartments are worth the same as unregulated apartments; restricted land is worth the same as unrestricted land;
restrictions can evaporate into thin air; a disclaimer by one party casting responsibility on another party exonerates the other party’s lies,” Engoron wrote in
his 35-page ruling. “That is a fantasy world, not the real world.”

Manhattan prosecutors had looked into bringing a criminal case over the same conduct but declined to do so, leaving James to sue Trump and seek penalties that
could disrupt his and his family’s ability to do business in the state.
</item 8>

<item 9>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
Mar-a-Lago Trump’s valuation (2021): $US612.1 million , Bloomberg valuation (2023): $US240 million, New York’s valuation (2021): $US27.6 million It’s fair to
say that Trump’s most famous property these days is no longer New York’s Trump Tower but his Mar-a-Lago Club, the historic Palm Beach estate built by Marjorie
Merriweather Post in the 1920s that he now calls home. New York’s valuation of Donald Trump’s Mar-a-Lago estate has raised eyebrows. Credit: AP The sprawling
resort has been at the centre of some publicised post-presidency scandals, from classified documents being kept in one of its bathrooms to the location where
Trump is said to have shared nuclear submarine secrets with Australian businessman Anthony Pratt — something that Trump denies.

It’s also a major focus in Trump’s fraud lawsuit. Trump made a series of agreements with Palm Beach County and the National Trust for Historic Preservation for
Mar-a-Lago to be used only as a club, giving up any development rights.
</item 9>

<item 10>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
They haven’t provided a more recent estimate.

Space inflation wasn’t the only issue. Between 2014 and 2015, Trump increased the value of the apartment to $US327 million from $US200 million. The prosecution
argues that was to mask a drop in the value of one of his other properties. “A discrepancy of this order of magnitude, by a real estate developer sizing up his
own living space of decades, can only be considered fraud,” Judge Arthur Engoron wrote in a September ruling. ‘The company has never been stronger and never
been better.’ Eric Trump Bloomberg values the property using recent Trump Tower sales, as well as those of comparable Manhattan penthouses. While some splashy
residences in much newer buildings on so-called Billionaires Row have listed for well in excess of $US100 million, they typically boast higher ceilings, more
up-to-date finishes, contemporary layouts, outdoor space and better views.
</item 10>
=== END ITEM id=4cf30d3baaef01b3 ===

=== ITEM id=4d60ec276e54a2f3 ===
Gold sentences and facts:
Sentence 0: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 0.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 0.1: The leading piece of drama returns.
Sentence 1: The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy Valley.
  Fact 1.0: The Big Ten suspended coach Jim Harbaugh for three games.
  Fact 1.1: The Big Ten suspended Jim Harbaugh on Friday.
  Fact 1.2: The team was on a plane en route to Penn State when the suspension happened.
  Fact 1.3: Michigan had hoped to block the suspension.
  Fact 1.4: The Wolverines (Michigan) will be without their coach on the sidelines in Happy Valley.

Retrieved items:
<item 1>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
🗣️ "Bet" pic.twitter.com/ENB7O8MyPv — The Sporting News (@sportingnews) November 10, 2023

Michigan announces intention to dispute suspension

Michigan has released a statement expressing frustration with the Big Ten's process and says it plans to seek a court order to block Harbaugh's suspension from
taking effect.

NEWS: Michigan says it intends to seek a court order with Jim Harbaugh preventing disciplinary action from taking effect.



Full statement: pic.twitter.com/p92yUCLsTX — Matt Fortuna (@Matt_Fortuna) November 10, 2023

Big Ten claims no evidence Harbaugh was aware of scheme

While the Big Ten says it's "proven" that an illegal sign-stealing scheme existed, the conference could not prove that Harbaugh knew about it. So, why was
Harbaugh suspended? The Big Ten asserts that, because he is the head coach, he has a responsibility for preventing this kind of scheme from taking place.
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Did Michigan really need Jim Harbaugh on Saturday?

Perhaps the result provides the short-term answer. No. 3 Michigan beat No. 10 Penn State 24-15 at Beaver Stadium on Saturday. The Wolverines threw one pass in
the second half – and that didn't even technically count – in a risk-averse game plan under acting coach Sherrone Moore. The defense smothered the Nittany
Lions, and that will lead to more buildup ahead of the showdown with No. 1 Ohio State on Nov. 25.

Then Moore let it all there in the Fox postgame interview with Jenny Taft. Even if the answer is no, Harbaugh was ever-present.

"I want to thank the Lord and I want to thank Coach Harbaugh," Moore said. "I f------ love you man. I love the s--- out of you, man. I did this for you, for
this university, the president, our AD.
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan noted that regarding evidence, there is nothing it is aware of to claim athletics personnel at Michigan committed the allegations, and said the rule
pertaining to electronic recordings of games is limited to the use of field equipment used by the institution for the game it is currently playing. It also said
Michigan has not seen much evidence, but rather has just heard a summary of second- and third-hand information.

The letter also said given the use of sign decoding by other universities, there is little reason to believe Stalions' efforts proved meaningful, and pointed
out the teams' average margin of victory since Stalions was suspended. It said the information obtained is "at most a reflection of information that teams can
and easily do obtain through public and permissible sources," and because of that it is "highly dubious" that Stalions' observations would have a "material
effect on the integrity of the competition."
</item 10>
=== END ITEM id=4d60ec276e54a2f3 ===

=== ITEM id=7e8583c38efe303d ===
Gold sentences and facts:
Sentence 0: It’s true that the Warriors received glowing grades from draft analysts following their selections of Kuminga, out of the G-League Ignite program, at No.
  Fact 0.0: It is true that the Warriors received glowing grades from draft analysts.
  Fact 0.1: The glowing grades followed the Warriors' selections of Kuminga.
  Fact 0.2: Kuminga came out of the G-League Ignite program.
  Fact 0.3: The Warriors selected Kuminga at No. (the sentence is cut off, so the pick number is not given).
Sentence 1: To win a bet on the Warriors, they need to win by 10 or more points.
  Fact 1.0: To win a bet on the Warriors, the Warriors need to win by 10 or more points.

Retrieved items:
<item 1>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
The Lakers won by a margin of 10 points. Since this is more than the 7-point spread, if you bet on the Lakers, you would win your wager.

How you lose a point spread bet

Say an NFL betting site has set the point spread for a Chicago Bears-Miami Dolphins game at the Bears -4. This indicates that the Bears are the favorites and
are expected to win by 4 points.

If you make a spread bet on the Bears, you're betting they'll win by more than 4 points.

However, let's assume the final score turns out to be Chicago Bears 24, Miami Dolphins 23. In this scenario, the Bears have won the game by only a 1-point
margin.

Since this is less than the 4-point spread, if you had the Bears, you would lose your wager. Despite the Bears winning the game, they failed to "cover" the
4-point spread, leading to a loss for those who wagered on them.

What is a push in a spread bet?
</item 1>

<item 2>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
However, there's also "stroke handicap" betting, akin to point spreads but with golf strokes.

For example, in a head-to-head bet between Tiger Woods and Jordan Spieth, if Woods is given a -2.5 stroke handicap as the favorite, he'd need to outplay Spieth
by 3 strokes or more to win the bet.

Betting on Spieth at +2.5 means he either wins outright or Woods wins by just 1 or 2 strokes.

Point spread betting FAQ

How does point spread betting work?

Point spread betting is a form of wagering where a sportsbook assigns a "spread" to level the playing field between two teams. One team, deemed the favorite,
will have points deducted, while the other team, the underdog, will have points added.

The favorite must win by more than the spread to "cover" and be considered a winning bet, while the underdog must either win outright or lose by fewer points
than the spread to be a winning bet.

What does the (-) in point spread betting mean?
</item 2>

<item 3>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Season Player Position Team 2023 Paolo Banchero Forward Orlando Magic 2022 Scottie Barnes Forward Toronto Raptors 2021 LaMelo Ball Guard Charlotte Hornets 2020
Ja Morant Guard Memphis Grizzlies 2019 Luka Dončić Guard/Forward Dallas Mavericks 2018 Ben Simmons Forward/Guard Philadelphia 76ers 2017 Malcolm Brogdon Guard
Milwaukee Bucks 2016 Karl-Anthony Towns Center Minnesota Timberwolves 2015 Andrew Wiggins Forward/Guard Minnesota Timberwolves 2014 Michael Carter-Williams
Guard Philadelphia 76ers 2013 Damian Lillard Guard Portland Trail Blazers 2012 Kyrie Irving Guard Cleveland Cavaliers 2011 Blake Griffin Forward Los Angeles
Clippers 2010 Tyreke Evans Guard/Forward Sacramento Kings

What other NBA futures can I bet on?

Rookie of the Year is not the only form of NBA futures betting.
</item 3>

<item 4>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
What is a push in a spread bet?

A "push" in betting refers to a situation where the final result of an event matches the set point spread, resulting in neither a win nor a loss for bettors.
When a push occurs, the wagers are typically refunded.

For instance, let's consider an NFL game where the Dallas Cowboys are favored by 7 points against the Philadelphia Eagles.

If the game ends with the Cowboys winning by exactly 7 points, those who bet on either team experience a push. Their wagers would be refunded as the Cowboys
neither exceeded nor fell short of the 7-point spread.

Sportsbooks often employ half-points in their lines to avoid this scenario. (They don’t want to return bets.) For example, if our point spread was Cowboys -7.5,
the game could not possibly end in a push.

Point spread betting by sport

Spread betting is popular across sports, but there can be slight differences. Let’s break it down.
</item 4>

<item 5>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Favorites to win the game are shown negative odds, such as -135, while the underdogs will have positive odds, like +165.

NBA over/under bets: Also known as totals. These bets are focused on the combined point total of the two teams playing in a game. (Winners and losers don’t
matter.) For example, the over/under could be set at 211.5 points in the Knicks vs. Heat matchup. If you were to pick the over on the bet, you’d need the two
teams to combine for 212 points or more to win. They need to score 211 points or fewer if you went with the under.

NBA props and futures

NBA player and team props are growing in popularity. Sportsbooks are taking note and providing more NBA props than ever.

After all, it’s much easier to focus on a single player and how he will perform than the entire team. Player props could focus on how many points a player
scores or how many rebounds he pulls down.
</item 5>

<item 6>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Point spread betting is a bet on the difference in the score between teams (or players) in a specific game or event. The favored team is provided a point spread
or handicap to try to equalize the odds with the less-favored team.

To place a point spread bet, select either the favored team or the underdog and predict if they will surpass or fall short of the given point spread. The odds
for both teams are set to try to even out the bets on both sides.

The best sports betting sites provide a daily menu of point spread bets in dozens of sports and events.

What is a point spread bet?

A point spread is a predetermined margin set by bookmakers to level the playing field between two teams.
</item 6>

<item 7>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Blackhawks bets win if they either triumph outright or lose by only one goal. If the score is Canadiens 3, Blackhawks 2, Blackhawks bets win since the
Canadiens' one-goal victory doesn't cover the 1.5-goal spread. A 4-1 win for the Canadiens would mean they covered the spread by a 3-goal margin.

See more: Tonight's NHL betting lines

Point spread odds in soccer

In a hypothetical MLS game between LA Galaxy and Atlanta United, LA Galaxy is set as a -1 favorite on the goal line. They'd need to win by 2+ goals for bets on
them to succeed. Atlanta United bets win if they either triumph outright or lose by a single goal.

With a score of LA Galaxy 2, Atlanta United 1, Atlanta bets win. However, a 3-1 win for LA Galaxy covers the spread with a 2-goal victory margin.
</item 7>

<item 8>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Point spread betting outcomes

When placing a point spread wager, there are a few potential outcomes:

The favorite can cover the spread by winning by a margin greater than the specified point spread

The underdog can either win the game outright or lose by fewer points than the spread, thereby "covering" the spread themselves

There's the possibility of a "push," where the difference in the game's score exactly matches the point spread, resulting in the return of the original bet to
all bettors

How to win a point spread bet

Suppose an NBA betting site has set the point spread for a Los Angeles Lakers-New York Knicks game at the Lakers -7.

This means the Lakers are the favorites and are expected to win by 7 points.

If you place a spread wager on the Lakers, to win your bet, the Lakers must win the game by more than 7 points.

Let's say the final score is Lakers 110, Knicks 100. The Lakers won by a margin of 10 points.
</item 8>

<item 9>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Moreover, information and public perception play a role. If a piece of news or a prevailing sentiment starts to gain traction about a particular team's
increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.

Point spread odds can and do change based on a combination of actual betting patterns, unexpected events, and shifts in public perception. Savvy bettors keep a
close eye on these line moves, as they can offer insights into where the value lies in a particular bet.
</item 9>

<item 10>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Then, you bet on whether:

The favored team will win by more than the spread

or

The underdog will lose by less than the spread (or just win outright)

In a point spread, the favorite is the team expected to win and is typically assigned a negative point value, indicating the number of points by which they need
to win for a bet on them to be successful.

The underdog is the team expected to lose and is typically assigned a positive point value, indicating the number of points they can lose by for a bet on them
to be successful.

If the underdog wins the contest outright, they’re also the spread winner.

What does betting ‘against the spread’ mean?

Most generally, betting against the spread just means placing a point spread bet.

You’re betting on the underdog to either win outright or lose by fewer points than the specified point spread, or you’re betting on the favorite to win by a
margin greater than the specified point spread.
</item 10>
=== END ITEM id=7e8583c38efe303d ===

=== ITEM id=639a2a6cbb7564fe ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: The friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: Making the friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: He later invited the 'Anti-Hero' singer on his podcast to one of his games at Arrowhead Stadium.
  Fact 1.1: The 'Anti-Hero' singer famously accepted the invitation on 24 September.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has shared that she has nothing to hide in her relationship with Kelce.
  Fact 2.2: Swift's relationship with Kelce is new.

Retrieved items:
<item 1>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 5>

<item 6>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 7>

<item 8>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 10>
=== END ITEM id=639a2a6cbb7564fe ===

=== ITEM id=9c63fd489e7d477c ===
Gold sentences and facts:
Sentence 0: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 0.0: The criminal trial of Sam Bankman-Fried was highly anticipated.
  Fact 0.1: Sam Bankman-Fried is the former CEO of FTX.
  Fact 0.2: FTX is a crypto exchange that is bankrupt.
  Fact 0.3: The criminal trial of Sam Bankman-Fried started on a Tuesday.
  Fact 0.4: The criminal trial determines whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 1: Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).
  Fact 1.0: Sam Bankman-Fried was compared to Warren Buffet.
  Fact 1.1: Many people called Sam Bankman-Fried the white horse of crypto.
  Fact 1.2: TechCrunch never called Sam Bankman-Fried the white horse of crypto.
Sentence 2: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 2.0: Sam Bankman-Fried, known as SBF, has pleaded not guilty.
  Fact 2.1: Sam Bankman-Fried has pleaded not guilty to some seven charges of fraud and conspiracy.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Sam Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense argued that the FTX founder acted in good faith.
  Fact 3.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 3.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 2>

<item 3>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 5>

<item 6>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 9>

<item 10>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 10>
=== END ITEM id=9c63fd489e7d477c ===
