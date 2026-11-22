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

=== ITEM id=22bb831a56727f33 ===
Gold sentences and facts:
Sentence 0: Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense, especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.
  Fact 0.0: Todd Boehly is American.
  Fact 0.1: Todd Boehly is a co-owner of Chelsea.
  Fact 0.2: Expanding Chelsea's profile and reach in the U.S. would make sense under Todd Boehly's co-ownership.
  Fact 0.3: Catarina Macario is a USWNT international.
  Fact 0.4: Mia Fishel is a USWNT international.
  Fact 0.5: Catarina Macario plays her club football in the U.S.
  Fact 0.6: Mia Fishel plays her club football in the U.S.
  Fact 0.7: CBS Sports holds WSL rights.
Sentence 1: It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again.
  Fact 1.0: The date was 24 September.
  Fact 1.1: Chelsea suffered a home Premier League defeat on 24 September.
  Fact 1.2: The 24 September defeat was Chelsea's second home Premier League defeat in a row.
  Fact 1.3: Chelsea's home Premier League defeat before the 24 September defeat was against Nottingham Forest.
  Fact 1.4: Chelsea were booed off again after the 24 September defeat.
Sentence 2: Late checks will be made on the fitness of Reece James, Axel Disasi and Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.
  Fact 2.0: Late fitness checks will be made on Reece James.
  Fact 2.1: Late fitness checks will be made on Axel Disasi.
  Fact 2.2: Late fitness checks will be made on Nicolas Jackson.
  Fact 2.3: Benoit Badiashile could return from a long-term injury.

Retrieved items:
<item 1>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

But the Swedish goalkeeper was at fault when Chelsea finally found the goal they were looking for, palming Gallagher’s cross after he had combined with Palmer
straight to the feet of Badiashile and he calmly sidefooted home.

Blackburn’s supporters then felt aggrieved when a Callum Brittain cross struck Gallagher’s trailing arm but once again Robinson waved away strong appeals for a
penalty.

The visitors showed they had no intention of surrendering without a fight when Harry Leonard caught Axel Disasi napping and raced onto a long ball at the start
of the second half, only to curl his shot wide of the post. Not for the first time, Robert Sánchez was then far too casual as the Chelsea goalkeeper’s pass was
almost intercepted.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 2>

<item 3>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
That is an understatement. And if there seemed a risk involved in swapping the stability of City for the seeming madhouse of Stamford Bridge, there was evidence
of more cool-headed thinking from a 21-year-old. Chelsea have been manic buyers but he said: “I did look at how many players were there but I don’t think people
realise how many players they actually got rid off, as well.”

The departures of, among others, Mason Mount, Kai Havertz, Christian Pulisic and Joao Felix, meant there could be opportunities in the attacking midfield
positions. “I looked at the squad and thought: ‘If I go there and know what I can do then I’ll have the chance of playing’,” he said. “And thankfully it’s
happened.” If there seemed plenty of traffic in the queue for places in Southgate’s attack, Palmer seems to have accelerated past several rivals. Now an England
debut could follow in the next week.
</item 3>

<item 4>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
“We should be proud of this performance.”

Blackburn certainly managed to cause problems for Chelsea even after the hosts made their intentions clear straight from the kick-off when Nicolas Jackson
forced a corner when his shot was blocked. Travis was very lucky not to concede a penalty when he slid in against Sterling after 10 minutes, with no VAR to
overrule what looked like a clear and obvious error from the referee Tim Robinson.

It took a good save from Leopold Wahlstedt to deny James but Blackburn then enjoyed a good period of possession in their opponents’ half. James was inches away
from breaking the deadlock midway through the first half but his shot just eluded the far post before Enzo Fernández forced Wahlstedt to palm away his effort
soon afterwards.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 4>

<item 5>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
It is no wonder Mauricio Pochettino spends most of his time watching Chelsea on the edge of the touchline with his hands clasped tightly in his pockets and a
tense expression on his face.

Even with his side 2-0 up and cruising against Blackburn thanks to goals from Benoît Badiashile and a rejuvenated Raheem Sterling, the Chelsea manager was a
bundle of nerves throughout what turned out to be a comfortable victory in the end.

But while the lack of cutting edge that has become a hallmark of the Todd Boehly-era in west London still looms large as he prepares for his first return to
table-topping Tottenham on Monday night, Pochettino could take heart from Cole Palmer’s brilliant performance as the former Manchester City forward helped to
dispatch Jon Dahl Tomasson’s inexperienced side.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 6>

<item 7>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 7>

<item 8>
Arteta reveals Pochettino influence on his career | Poch: He is family (Sky Sports, 2023-10-20T21:30:00+00:00)
Havertz will command particular focus after his early struggles at Arsenal. With just one goal in 12 appearances for the club there are questions about his form
and role.

Please use Chrome browser for a more accessible video player Ahead of Kai Havertz's return to Stamford Bridge, Sam Blitz takes a forensic look at his tactical
role at Arsenal

Armando Broja will miss Chelsea's game against Arsenal due to "irritation" in his knee. Late checks will be made on the fitness of Reece James, Axel Disasi and
Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.

Enzo Fernandez and Moises Caicedo should be available despite travelling to South America during the international break.

Ben Chilwell, Carney Chukwuemeka, Christopher Nkunku, Romeo Lavia, Wesley Fofana and Trevoh Chalobah remain unavailable.

Arsenal will make a late call on the fitness of Bukayo Saka, William Saliba and Leandro Trossard.
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 9>

<item 10>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
Arsenal represent an acid test of the progress and the start of a daunting run in the league; after them, Chelsea face Brentford, Tottenham, Manchester City,
Newcastle, Brighton and Manchester United. Pochettino, though, pushed back against the notion that he was seeking a first statement win at the club, something
to ignite his tenure. All he wants are points to drive his team up from 11th in the table.

Pochettino has appeared mired in a selection crisis since his arrival at Chelsea; James only one element of it and he will be without seven others through
injury against Arsenal.

“The moment we have all the squad at a high level and fit, we can say: ‘OK, we are going to Arsenal, Manchester City and we are going to beat you.’ At the
moment, with the circumstances, I don’t think this game is going to be a statement to say: ‘Be careful, Chelsea is coming,’ or something like this.”
</item 10>
=== END ITEM id=22bb831a56727f33 ===

=== ITEM id=3fa7f1f6febe2255 ===
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
Start by clicking on our exclusive DraftKings promo code link

Register as a new user by supplying personal info including your name, email, last four of your SSN#, and mailing address

Deposit $5 to activate $50 bonus bet

Bet $5 and Get $150 in bonus bets

To max out DraftKings' deposit match at $1,000, make a minimum deposit of $5,000 and meet the 25X playthrough requirement.

Claim here: DraftKings promo code for $1,200 in bonus bets

BetRivers promo code SPORTSRIV unlocks 2nd chance bet for CBK games including Duke vs. Arkansas

The BetRivers promo code SPORTSRIV gives new users a 2nd chance bet up to $500 depending on your state.

If your first bet settles as a loss, BetRivers will refund your account in bonus bets.
</item 1>

<item 2>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Bet CBK games here: $1,550 in bonus bets with BetMGM bonus code TSN50

Bet365 bonus code TSNCO for First Bet Safety Net up to $1,000 or Bet $5, Get $150 in bonus bets for ACC-SEC Challenge games

New customers at Bet365 have their choice of promo offers when using Bet365 bonus code TSNCO. The First Bet Safety Net protects your initial wager up to $1,000!

You can also choose the Bet $5, Get $150 in bonus bets offer. Both of these promos can be used to bet on any CBK games including tonight's ACC-SEC Challenge
matchups!

Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.
</item 2>

<item 3>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
7 Tennessee vs. No. 14 North Carolina. Duke vs. Arkansas is another marquee matchup on tonight's slate of CBK games.

Claim thousands in sports betting bonuses for college basketball games with these promo codes and links

Top-rated sportsbooks -- BetMGM, Bet365, FanDuel, Caesars Sportsbook, DraftKings and BetRivers -- make it worth your while when signing up as a new user to the
tune of $5K+ in bonus bets!

Click our links to activate the best sports betting bonuses in the industry and register your sportsbook accounts now!

Caesars Sportsbook promo code NEWS1000 for $1,000 First Bet on Caesars for Volunteers vs. Tar Heels

Caesars Sportsbook promo code NEWS1000 gives new users a First Bet up to $1,000 on Caesars!
</item 3>

<item 4>
2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet (CBSSports.com, 2023-10-13T21:17:17+00:00)
4. BetMGM Sportsbook: The signup offer at BetMGM caters more to larger players, a $1500 risk free offer which will credit a customer with bonus bets in the
event their first real money wager loses. The boosts and risk free offers on the site cater to smaller players as the limits for those tokens are generally less
than the other sites. With a friendly interface and a rewards program that can be redeemed for bonus bets or MGM Resort credit, this sportsbook is one of the
best in the business for a reason.

5. Caesars Sportsbook: Similar to MGM, the rewards program is what Caesars is most known for in that customers can redeem their wagering credits for use at
Caesars properties. New customers can get up to $1000 in bonus bets if they lose their first bet. The sportsbook offers an extensive betting menu and
specializes in specific boosts related to your home state teams.

6.
</item 4>

<item 5>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking our Caesars Sportsbook promo code link and sign up for a new account. If your first bet loses, Caesars will credit your account with bonus
bets in the amount of your wager up to $1,000.

Click to claim: Caesars Sportsbook promo code NEWS1000 for First Bet up to $1K on Caesars

BetMGM bonus code TSN50 unlocks $1,550 in bonus bets for Blue Devils vs. Razorbacks

Use our exclusive BetMGM bonus code TSN50 to claim a First Bet bonus of up to $1,500, and a $50 bonus bet for customers in select states.

Click our BetMGM bonus code link and register for your new account with promo code TSN50 to unlock this bonus.
</item 5>

<item 6>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
This offer requires a minimum $10 deposit, but you can deposit up to the amount of your first bet to $1,500 to maximize your potential winnings. If your first
bet on the app loses, BetMGM will refund your account with bonus bets in the amount of the original wager up to $1,500.

The extra $50 bonus bet offer is not available in Washington D.C., Kentucky, Mississippi, Nevada, New York, Puerto Rico and Ontario. Customers in those areas
can use BetMGM bonus code SPORTSPICK for the $1,500 First Bet offer.
</item 6>

<item 7>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
To claim this top-notch welcome offer, click our Caesars Sportsbook promo code link and sign up for a new Caesars account. Be sure Caesars Sportsbook promo code
NEWS1000 is entered when prompted. Then make your first deposit ($10 minimum).

After that, place your first Caesars Sportsbook bet and if it loses you’ll get the amount of that wager (up to $1,000) as a single bonus bet.

Get up to $1,200 in bonuses for Eagles vs.
</item 7>

<item 8>
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
</item 8>

<item 9>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.

Claim here: Bet365 bonus code TSNCO for promo of your choice

FanDuel promo code for Bet any $5 moneyline, Get $150 in bonus bets if your team wins in CBK game

FanDuel has a new offer when using our exclusive FanDuel promo code link. Bet $5 on any moneyline, and Get $150 in bonus bets if your team wins!

It's important to note that there are no odds restrictions, so it'd be prudent to bet a big favorite over an outmatched underdog on the CBK schedule. You need
to win the wager to receive $150 in bonus bets.
</item 9>

<item 10>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
DraftKings | BetMGM vs. FanDuel | Caesars vs. BetMGM
</item 10>
=== END ITEM id=3fa7f1f6febe2255 ===

=== ITEM id=a816faa0a26b466e ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made big improvements with its new iteration.
  Fact 0.2: The minor physical upgrades are part of the new iteration alongside the big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: Valve looks at a store that could be both.
  Fact 1.1: Valve decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units ready to ship on November 16th.

Retrieved items:
<item 1>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 4>

<item 5>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I do not recommend licking the screen.)

Battery life

One of the downsides of the first Steam Deck was its short battery life, which maxed out at eight hours, and that was only under ideal conditions. Valve says
the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and it charges faster — the company claims it will go from 20
percent to 80 percent in 45 minutes.

Custom AMD APU

The Steam Deck OLED has a die-shrunk version of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process
Sony takes advantage of to build the PS4 Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger
battery and larger fan to its existing chassis.
</item 5>

<item 6>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 7>

<item 8>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 8>

<item 9>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 9>

<item 10>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 10>
=== END ITEM id=a816faa0a26b466e ===

=== ITEM id=f23c97deb8290d4e ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

Retrieved items:
<item 1>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 1>

<item 2>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 4>

<item 5>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I think the philosophy change would do both of those companies much good.

If you get your way in terms of being able to freely put your own store on Android, do you believe that would only be a game store or would it also be an app
store? We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.

So the Epic Games Store isn’t a games store, right? It’s the store operated by Epic Games. So we have a lot of non-games there already. We have the Brave web
browser, we have a number of software creation tools including Unreal Engine, and there’s more coming, including some other awesome creation tools and
productivity tools. We’ll host any app anybody wants of any sort.

I think the gaming market is something we’re uniquely close to, and so I think we would likely be able to forge closer partnerships and opportunities in gaming,
but we’ll be open to everybody on Android as we are on PC.
</item 5>

<item 6>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 7>

<item 8>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 8>

<item 9>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 9>

<item 10>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 10>
=== END ITEM id=f23c97deb8290d4e ===

=== ITEM id=cb9ef8b3074b0966 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made big improvements with its new iteration.
  Fact 0.2: The minor physical upgrades are part of the new iteration alongside the big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: Valve looks at a store that could be both.
  Fact 1.1: Valve decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units ready to ship on November 16th.

Retrieved items:
<item 1>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 4>

<item 5>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I do not recommend licking the screen.)

Battery life

One of the downsides of the first Steam Deck was its short battery life, which maxed out at eight hours, and that was only under ideal conditions. Valve says
the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and it charges faster — the company claims it will go from 20
percent to 80 percent in 45 minutes.

Custom AMD APU

The Steam Deck OLED has a die-shrunk version of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process
Sony takes advantage of to build the PS4 Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger
battery and larger fan to its existing chassis.
</item 5>

<item 6>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 7>

<item 8>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 8>

<item 9>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 9>

<item 10>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 10>
=== END ITEM id=cb9ef8b3074b0966 ===

=== ITEM id=cfd3d166a95e875e ===
Gold sentences and facts:
Sentence 0: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 0.0: Swift has shared that she has nothing to hide.
  Fact 0.1: Swift is in a new relationship with Kelce.
  Fact 0.2: Swift has shared that she has nothing to hide in her new relationship with Kelce.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: The podcast host, referred to as 'he', later invited the 'Anti-Hero' singer to one of his games at Arrowhead Stadium.
  Fact 1.1: The invitation was to a game at Arrowhead Stadium.
  Fact 1.2: The 'Anti-Hero' singer accepted the invitation.
  Fact 1.3: The 'Anti-Hero' singer famously accepted the invitation on 24 September.

Retrieved items:
<item 1>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.
</item 1>

<item 2>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 2>

<item 3>
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
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 5>

<item 6>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
</item 6>

<item 7>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 7>

<item 8>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 8>

<item 9>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 9>

<item 10>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 10>
=== END ITEM id=cfd3d166a95e875e ===
