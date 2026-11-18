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

This file has 6 items.

=== ITEM id=8fe0226d8da4bb22 ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet with his number on it.
  Fact 0.1: Friendship bracelets are a fan tradition at her concerts.
  Fact 0.2: He made the friendship bracelet to give to her.
  Fact 0.3: The friendship bracelet seemingly expresses his interest in Swift.
Sentence 1: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 1.0: Swift has shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 2>

<item 3>
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
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 4>

<item 5>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 5>

<item 6>
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
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 7>

<item 8>
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
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Head coach Andy Reid was also asked about his star tight end's supposed relationship, to which he quipped, "I set them up."

Sept. 24: Kelce and Swift leave stadium together

"You know when it's time to go." - It's Time To Go

This was the first time the two were photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift
wearing a Chiefs zip-up.

Sept. 24: The two leave in a convertible

"Nothing good starts in the getaway car." - Getaway Car

Swift and Swift were spotted in a convertible, with the tight end driving, after the Chiefs' win. Kelce reportedly rented out a restaurant for a private party
with Swift and his teammates.

"Taylor Swift likes that Travis Kelce 'pursued her' and the two are having a 'great time getting to know each other,'" a source told Entertainment Tonight.
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 10>
=== END ITEM id=8fe0226d8da4bb22 ===

=== ITEM id=aa5263c1fa1d1bb5 ===
Gold sentences and facts:
Sentence 0: New customers can get up to $1000 in bonus bets if they lose their first bet.
  Fact 0.0: New customers can get up to $1000 in bonus bets.
  Fact 0.1: New customers can get bonus bets if they lose their first bet.
Sentence 1: We expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses.
  Fact 1.0: Caesars is expected to offer the same offer in VT as it has elsewhere.
  Fact 1.1: New sign-ups are eligible to receive up to a $1,000 Bonus Bet.
  Fact 1.2: New sign-ups are eligible for the Bonus Bet if their first bet loses.

Retrieved items:
<item 1>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
This offer requires a minimum $10 deposit, but you can deposit up to the amount of your first bet to $1,500 to maximize your potential winnings. If your first
bet on the app loses, BetMGM will refund your account with bonus bets in the amount of the original wager up to $1,500.

The extra $50 bonus bet offer is not available in Washington D.C., Kentucky, Mississippi, Nevada, New York, Puerto Rico and Ontario. Customers in those areas
can use BetMGM bonus code SPORTSPICK for the $1,500 First Bet offer.
</item 1>

<item 2>
2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet (CBSSports.com, 2023-10-13T21:17:17+00:00)
4. BetMGM Sportsbook: The signup offer at BetMGM caters more to larger players, a $1500 risk free offer which will credit a customer with bonus bets in the
event their first real money wager loses. The boosts and risk free offers on the site cater to smaller players as the limits for those tokens are generally less
than the other sites. With a friendly interface and a rewards program that can be redeemed for bonus bets or MGM Resort credit, this sportsbook is one of the
best in the business for a reason.

5. Caesars Sportsbook: Similar to MGM, the rewards program is what Caesars is most known for in that customers can redeem their wagering credits for use at
Caesars properties. New customers can get up to $1000 in bonus bets if they lose their first bet. The sportsbook offers an extensive betting menu and
specializes in specific boosts related to your home state teams.

6.
</item 2>

<item 3>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
The site offers frequent odds boosts and other bonuses, and new sign-ups also get to join the famed Caesars Rewards program that can earn them more site bonuses
among other benefits.

Here are details explaining how the Caesars Sportsbook welcome offer works:

When registering your new Caesars Sportsbook account, enter the bonus code NEWS1000 .

. Place a first bet within 30 days and if it loses, receive back the equivalent amount in the form of a Bonus Bet worth up to $1,000 .

. After that first bet has been settled, you’ll see the Bonus Bet in your account within two days .

. Use your Bonus Bet within 14 days or else it will expire.

or else it will expire. If you win with your Bonus Bet, you only receive the winnings and not the bet amount back.

BetRivers Vermont promo code

BetRivers Sportsbook is another site that has launched in many states and could be among those coming to Vermont. BetRivers has a lot to offer and many have
found it a great choice elsewhere.
</item 3>

<item 4>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Bet CBK games here: $1,550 in bonus bets with BetMGM bonus code TSN50

Bet365 bonus code TSNCO for First Bet Safety Net up to $1,000 or Bet $5, Get $150 in bonus bets for ACC-SEC Challenge games

New customers at Bet365 have their choice of promo offers when using Bet365 bonus code TSNCO. The First Bet Safety Net protects your initial wager up to $1,000!

You can also choose the Bet $5, Get $150 in bonus bets offer. Both of these promos can be used to bet on any CBK games including tonight's ACC-SEC Challenge
matchups!

Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.
</item 4>

<item 5>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Use our exclusive sports betting promo codes to unlock thousands of dollars in bonus bets for all of this week's CBK games (offers may vary among states):

Top Sportsbook Bonuses Offer Promo Code Caesars Sportsbook Get $1,000 First Bet on Caesars NEWS1000 BetMGM $1,500 First Bet + $50 bonus bet TSN50 Bet365 First
Bet Safety Net up to $1,000 OR Bet $5, Get $150 in bonus bets TSNCO FanDuel Bet $5 on any moneyline, Get $150 in bonus bets if your team wins Click to claim
DraftKings Up to $1,200 in welcome offers Click to claim BetRivers 2nd chance bet up to $500 SPORTSRIV

The ACC-SEC Challenge continues today with seven games starting with No. 7 Tennessee vs. No. 14 North Carolina. Duke vs.
</item 5>

<item 6>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
7 Tennessee vs. No. 14 North Carolina. Duke vs. Arkansas is another marquee matchup on tonight's slate of CBK games.

Claim thousands in sports betting bonuses for college basketball games with these promo codes and links

Top-rated sportsbooks -- BetMGM, Bet365, FanDuel, Caesars Sportsbook, DraftKings and BetRivers -- make it worth your while when signing up as a new user to the
tune of $5K+ in bonus bets!

Click our links to activate the best sports betting bonuses in the industry and register your sportsbook accounts now!

Caesars Sportsbook promo code NEWS1000 for $1,000 First Bet on Caesars for Volunteers vs. Tar Heels

Caesars Sportsbook promo code NEWS1000 gives new users a First Bet up to $1,000 on Caesars!
</item 6>

<item 7>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking on our exclusive DraftKings promo code link

Register as a new user by supplying personal info including your name, email, last four of your SSN#, and mailing address

Deposit $5 to activate $50 bonus bet

Bet $5 and Get $150 in bonus bets

To max out DraftKings' deposit match at $1,000, make a minimum deposit of $5,000 and meet the 25X playthrough requirement.

Claim here: DraftKings promo code for $1,200 in bonus bets

BetRivers promo code SPORTSRIV unlocks 2nd chance bet for CBK games including Duke vs. Arkansas

The BetRivers promo code SPORTSRIV gives new users a 2nd chance bet up to $500 depending on your state.

If your first bet settles as a loss, BetRivers will refund your account in bonus bets.
</item 7>

<item 8>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking our Caesars Sportsbook promo code link and sign up for a new account. If your first bet loses, Caesars will credit your account with bonus
bets in the amount of your wager up to $1,000.

Click to claim: Caesars Sportsbook promo code NEWS1000 for First Bet up to $1K on Caesars

BetMGM bonus code TSN50 unlocks $1,550 in bonus bets for Blue Devils vs. Razorbacks

Use our exclusive BetMGM bonus code TSN50 to claim a First Bet bonus of up to $1,500, and a $50 bonus bet for customers in select states.

Click our BetMGM bonus code link and register for your new account with promo code TSN50 to unlock this bonus.
</item 8>

<item 9>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
Place your first bet and, if it loses, BetRivers will refund the amount of the wager in bonus bets worth up to $100, $250 or $500, depending on your state.

Use our BetRivers bonus code SPORTSRIV and you’ll receive a Second Chance Bet worth up to $100 (for users in AZ, IA, OH and WV), $250 (available in CO and NJ)
or $500 (if betting from IL, IN, LA, MD, MI, PA and VA).

Caesars Sportsbook promo code NEWS1000 offers $1,000 First Bet On Caesars for Eagles vs. Seahawks MNF

Use our Caesars Sportsbook promo code NEWS1000 to register a new Caesars account and claim a First Bet bonus of up to $1,000.
</item 9>

<item 10>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
Top Sportsbook Bonuses Offer Promo Code Caesars Sportsbook Get $1,000 First Bet on Caesars NEWS1000 BetMGM $1,500 First Bet SPORTSPICK Bet365 First Bet Safety
Net up to $1,000 --or-- Bet $5, Get $150 in bonus bets TSNCO FanDuel Bet $5 on any moneyline, Get $150 in bonus bets if your team wins Click to claim DraftKings
Up to $1,200 in welcome offers Click to claim BetRivers 2nd chance bet up to $500 SPORTSRIV

Claim $5,000+ in sports betting bonuses for Eagles vs.
</item 10>
=== END ITEM id=aa5263c1fa1d1bb5 ===

=== ITEM id=b0c27a8032df7080 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.1: Caroline Ellison was the CEO of Alameda.
  Fact 0.2: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.3: The government claimed Bankman-Fried had secret access to customer money.
  Fact 0.4: The government claimed the customer money included both cash and crypto.
Sentence 1: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 1.0: Bankman-Fried is known as SBF.
  Fact 1.1: Bankman-Fried has pleaded not guilty to some seven charges of fraud and conspiracy.
Sentence 2: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 2.0: Judge Lewis Kaplan intervened to ask whether Bankman-Fried had ever been told by Yedidia about that money, in words or in substance.
  Fact 2.1: Bankman-Fried admitted he had been told about that money only after Judge Lewis Kaplan intervened.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense argued that Bankman-Fried, the FTX founder, acted in good faith.
  Fact 3.2: The defense argued that Bankman-Fried, the FTX founder, never meant to commit fraud or steal.
  Fact 3.3: The defense argued that Bankman-Fried, the FTX founder, basically got in over his head.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 4>

<item 5>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 6>

<item 7>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
It is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet, and the judge has already
instructed him to answer the question he’s being asked by his own lawyer.

The jury is watching all of this intently.

The main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.

Also, sometimes when Bankman-Fried says “we,” he only means himself.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud
“from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer
funds.” Follow along for all the latest news and regular updates from the trial.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 10>
=== END ITEM id=b0c27a8032df7080 ===

=== ITEM id=d211e8c9b77ffab5 ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.0: A point that will probably come up later in the trial is one Epic did not emphasize on day one.
  Fact 0.1: Epic chose to focus on painting Google as the bad guy on day one.
  Fact 0.2: Epic chose to focus more on simply painting Google as the bad guy on day one.
Sentence 1: “Certainly there wasn’t a valid alternative to Google at the time,” Cue said.
  Fact 1.0: Cue said there certainly was not a valid alternative to Google at the time.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
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
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 5>

<item 6>
Graph query row:
company: Google
Epic_case: Tim Sweeney on Epic’s victory royale over Google
Apple_search_defense: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’
publishers_suit: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 10>
=== END ITEM id=d211e8c9b77ffab5 ===

=== ITEM id=d8b53c305eb1dedc ===
Gold sentences and facts:
Sentence 0: “But nah, it was awesome.” On Saturday, the competition also said goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season, and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.
  Fact 0.0: A speaker said 'But nah, it was awesome.'
  Fact 0.1: On Saturday, the competition said goodbye to a range of players.
  Fact 0.2: On Saturday, the competition said goodbye to Western Bulldogs forward Richelle Cranston.
  Fact 0.3: Richelle Cranston has battled stage five chronic kidney disease while playing this season.
  Fact 0.4: On Saturday, the competition said goodbye to Hawthorn's Akec Makur Chuot.
  Fact 0.5: Akec Makur Chuot is a Hawthorn player.
  Fact 0.6: Akec Makur Chuot played 40 games for three clubs.
Sentence 1: The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year.
  Fact 1.0: The Dogs are Richelle Cranston's third club.
  Fact 1.1: Richelle Cranston started at Melbourne.
  Fact 1.2: Richelle Cranston had a stint with Geelong.
  Fact 1.3: Richelle Cranston joined the Bulldogs last year.

Retrieved items:
<item 1>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 1>

<item 2>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Dogs forward reveals battle with kidney disease Western Bulldogs forward Richelle Cranston has revealed she’s been living, and playing, with stage five chronic
kidney disease. “I still manage to do the things I love,” the inaugural AFLW player posted to Instagram. “It’s just really hard sometimes.” Richelle Cranston of
the Bulldogs: “It’s just really hard sometimes.” Credit: Getty Images The 34-year-old posted the caption along with a video detailing her life with the disease
including weekly blood tests and daily blood pressure checks, tracking water and food intake, taking medications including resonium (to lower potassium levels),
all while training in the high-performance environment.

The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year. Cranston lined up for
the Bulldogs in her side’s 57-point loss to Sydney on Saturday at Whitten Oval, 14.12 (96) to 6.3 (39).
</item 2>

<item 3>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Tackled above waist and below shoulders. No sling or double motion. Sheppard lands on her back. ′ ‘Girls’ play hard too,” Harford said on X, the social media
site formerly known as Twitter. This echoed Collingwood defender and Fox Footy commentator Ruby Schleicher’s thoughts at the time, who called it a “monster
tackle” but believed it was “fair”. “I don’t see any dirtiness in that,” Schleicher said on Fox Footy. “It was pretty heavy on her head, wasn’t it? I look at
that – I still see a fair tackle. There’s no doubt the match review [officer] will look at it because they have to I guess. The duty of care for the player.”

The decision a week comes after Adelaide premiership defender Najwa Allen was handed a three-ban suspension, the longest ban in AFLW history, for a high bump on
Western Bulldog Kirsten McLeod. The Crows elected not to challenge the decision.
</item 3>

<item 4>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I’m excited like I think the group’s coming together really nicely and where really well-prepared compared to previous years,” Riddell told Fox Footy. “So, I
can’t wait I think it’s an exciting time for us.” The game was one-way traffic against the Dogs with the game played almost entirely in the Roos’ attacking
half. Riddell credited her side’s “team performance” after two losses to the Dees and Crows. North Melbourne have secured a top-four spot for finals. Credit:
Getty For the Western Bulldogs, it was another tough day out as they notched their ninth loss of the season. They won just one game this year, an eight-point
victory last weekend over 17th-placed West Coast.

Britney Gutknecht kicked the Dogs’ only goal against the Kangaroos, which came early in the final term to avoid a record for the lowest AFLW score in the
history of the competition.
</item 5>

<item 6>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.
</item 6>

<item 7>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Bridie Hipwell kicked the only goal of the third quarter to give Hawthorn a game-high nine-point lead. Loading Richmond rebounded in the last term, with
Courtney Jones ending Hawthorn’s three-goal run with her first touch of the night, cutting the margin to three points. Dogged by inaccuracy in front of goal,
Yassir reduced the deficit to a single point before finally breaking through on her third attempt. That put the Tigers ahead by six points with 11 minutes left,
before Caitlin Greiser sealed the win with a massive goal from 45 metres.
</item 7>

<item 8>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 8>

<item 9>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“It was an awesome game and four quarters of Port Adelaide footy … It was awesome to finish up like that,” Phillips told Fox Footy after the 13.10 (88) to 4.6
(30) win over the Giants. Erin Phillips celebrates a goal during her final AFLW match. Credit: AFL Photos Earlier, she told the ABC that she wanted her legacy
to include doing everything she could to grow the game.

“Hopefully I can be remembered as someone who took their opportunities, loved the game and was willing to do whatever it took to ensure the next generation was
in a better position,” she said. Richelle Cranston of the Bulldogs is chaired off the ground by teammates Isabelle Pritchard and Ellie Blackburn after
announcing her retirement. Credit: Getty The 38-year-old kicked a momentous goal from a set shot in the final quarter, and she was smothered by her jubilant
teammates afterwards.
</item 9>

<item 10>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I don’t think I’ve been as nervous to kick a goal in my life, to be honest,” she joked. “But nah, it was awesome.” On Saturday, the competition also said
goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season,
and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.

Cranston, who was an inaugural AFLW player, will start kidney dialysis in January so had no option but to hang up the boots and focus on her treatment and
health. “[I’m feeling] sad. Emotions hit me when the siren went. It’s been a really emotional week, the girls have really gotten around me. But yeah, my health
has sort of deteriorated, it’s been a really hard year,” Cranston told Fox Footy after the Bulldogs lost to North Melbourne 7.13 (55) to 1.3 (9).
</item 10>
=== END ITEM id=d8b53c305eb1dedc ===

=== ITEM id=c64b31245d954ba5 ===
Gold sentences and facts:
Sentence 0: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 0.0: Kenneth Walker III plays for the Seahawks.
  Fact 0.1: Kenneth Walker III took four carries against the Rams.
  Fact 0.2: Kenneth Walker III took four carries against the Rams before leaving.
  Fact 0.3: Kenneth Walker III left with an oblique injury.
Sentence 1: Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging injuries.
  Fact 1.0: Entering Week 12, Tee Higgins failed to suit up due to a nagging hamstring injury.
  Fact 1.1: Entering Week 12, Noah Brown failed to suit up due to a nagging knee injury.
  Fact 1.2: Entering Week 12, Treylon Burks failed to suit up due to a nagging concussion.
  Fact 1.3: Entering Week 12, Kadarius Toney failed to suit up due to a nagging ankle injury.
  Fact 1.4: Entering Week 12, Kadarius Toney failed to suit up due to a nagging hip injury.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 3>

<item 4>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 4>

<item 5>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.
</item 5>

<item 6>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 7>

<item 8>
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
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 9>

<item 10>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 10>
=== END ITEM id=c64b31245d954ba5 ===
