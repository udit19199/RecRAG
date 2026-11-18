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

=== ITEM id=39c4f5cf6ea43e82 ===
Gold sentences and facts:
Sentence 0: The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.
  Fact 0.0: The agreement meant that Apple never switched to a Google competitor.
  Fact 0.1: The agreement meant that Apple never allowed users to choose their browser when setting up their iPhones.
Sentence 1: Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months.
  Fact 1.0: Apple has already indicated that it plans to appeal.
  Fact 1.1: Appealing is a lengthy process.
  Fact 1.2: Appealing can take around 18 months.
Sentence 2: They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.
  Fact 2.0: Google uses its store.
  Fact 2.1: Google uses its payments.
  Fact 2.2: Google forces developers to all have the same terms.
  Fact 2.3: Google forces OEMs to all have the same terms.
  Fact 2.4: Google forces carriers to all have the same terms.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> US v. Google

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, App Tracking Transparency, Cue’s testimony, location
tracking, Mozilla, Google Search, Apple, Maps, data collection, Samsung, Google, search, Justice Department
</item 1>

<item 2>
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
</item 2>

<item 3>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Earlier this year, the US International Trade Commission (ITC) ruled that Apple had infringed on two patents from medical device maker Masimo. As a result, the
ITC said it would impose an import ban on the Apple Watch Series 9 and Ultra 2 starting December 26th. At the time it was hard to believe that this would
actually come to pass: Apple gets sued all the time, and even when it loses, how often does it actually face dramatic consequences?

Well, now would be one of those times. Experts say that, barring a Christmas miracle, it’s unlikely that Apple will find a way to escape the ban. Case in point,
the company shocked everyone yesterday when it decided to preemptively pull the watches from its online store starting December 21st at 3PM ET. And after the
24th, they’ll disappear from Apple Stores, too.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, US International Trade Commission, Ultra 2, import ban, patent infringement, Apple online store, Apple Stores, Apple Watch removal from Apple’s
online store, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, Apple Watch Series 9, The Verge
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, ISA, John Schmidtlein, Google / Apple partnership,
Google Search, Apple, Google, Safari, The Verge
</item 4>

<item 5>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Apple getting a presidential veto would be like lightning striking the same place twice.

“It is extraordinarily rare for the President to overturn a decision from the International Trade Commission and in my view, [it’s] unlikely in this particular
case,” says Andrei Iancu, co-chair of the Council for Innovation Promotion (C4IP) and partner at Sullivan & Cromwell LLP, who was also the director of the US
Patent and Trademark Office from 2018 to 2021.

To get a veto, a company usually has to show a public interest or health policy basis — and that isn’t the case here. “It’s unlikely that the import exclusion
order will be disapproved by the USTR and the president,” says Smith Brittingham, partner at Finnegan, Henderson, Farabow, Garrett & Dunner, LLP. “The only real
answer would be, ‘Well, it’s Apple.’”

That said, Apple did manage to beat the system once. In 2013, then President Barack Obama vetoed an import ban on the iPhone.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Why the Apple Watch is being banned — and how Apple can avoid it, International Trade Commission, Andrei Iancu, Council for Innovation
Promotion, Sullivan & Cromwell LLP, US Patent and Trademark Office, Smith Brittingham, Finnegan, Henderson, Farabow, Garrett & Dunner, LLP, USTR, Barack Obama,
veto of an import ban on the iPhone, public interest, health policy, Apple Watch
</item 5>

<item 6>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
“From a legal perspective, whatever change Apple makes has to be significant enough to avoid the patent, and it cannot be what’s called a ‘colorable’
difference. Minor changes will not be sufficient,” Iancu says.

Basically, Apple can’t simply make a small tweak to the code and call it a day. Masimo claims that the patents are hardware related, so even completely
disabling the sensor may not work depending on how the patent is written. Apple could remove it entirely from the design, but that would require mucking around
with Apple’s meticulous supply chain and could take some time.

Then there are the options that would cost Apple some money. Apple could sit down with Masimo and hammer out a licensing deal for the infringed patents. In a
New York Times interview, Masimo CEO Joe Kiani said Apple “had not engaged in licensing negotiations.” It’s an easy-seeming fix, so why not do it? For starters,
both parties have to want to settle.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, Joe Kiani, patents, Iancu, New York Times, The Verge
</item 6>

<item 7>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
For starters, both parties have to want to settle.

“Apple is a tough nut to crack,” says Brittingham, noting that Apple is one of the most sued companies in Silicon Valley. The company’s large cash reserves also
make taking Apple to court a costly endeavor. Masimo, for example, has reportedly spent $60 million litigating the issue thus far. “I think they tend to hold
out as long as possible in order to make it less appetizing to sue Apple.”

The other option is for Apple to avoid the import ban altogether. Iancu says Apple could simply decide to manufacture the watches within the US instead of
overseas — but that’s a significant supply chain change that’s unlikely to happen any time soon.

Does this set a precedent?

This import ban isn’t the only one Apple is facing. In February, the Biden administration also declined to veto another ITC import ban and cease-and-desist
order stemming from a dispute with AliveCor over the Apple Watch’s EKG features.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, US, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, Brittingham, Iancu, ITC, EKG features, Apple Watch
import ban, ITC import ban and cease-and-desist order, Dispute with AliveCor over Apple Watch EKG features, Litigating the issue, AliveCor, Silicon Valley
</item 7>

<item 8>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And after the 24th, they’ll disappear from Apple Stores, too.

“I think Apple sees the writing on the wall and they’re preparing themselves,” says Ben Levi, a partner at Levi Snotherly & Schaumberg, which has experience
litigating ITC disputes.

It’s rare to see Big Tech lose — and with such tangible consequences at that. There are still ways that Apple can keep the watches on sale, but it’ll likely
take a lot of waiting or a lot of money to make it happen.

How Apple got here

This ban is the result of a long-standing patent dispute between Apple and the medical device maker Masimo. The latter is known for its pulse oximetry tech,
generally referred to as SpO2 or blood oxygen saturation in the wearable world, and it claims that Apple is using that patented technology without permission.

This particular story started about 10 years ago when Apple reached out to Masimo about a potential partnership around blood oxygen features on its wearables.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Ben Levi, Levi Snotherly & Schaumberg, pulse oximetry tech, patent dispute between Apple and Masimo, using patented technology without
permission, Apple Stores, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, ITC, The Verge
</item 8>

<item 9>
Graph query row:
article: Why the Apple Watch is being banned — and how Apple can avoid it
publisher: The Verge
relationship: MENTIONS
connected_entity: Apple
</item 9>

<item 10>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
In 2013, then President Barack Obama vetoed an import ban on the iPhone. At the time, Samsung accused Apple of infringing on its cellular data patents. But in
that case, according to Levi, Apple was able to make the argument that there were issues regarding fair, reasonable, and nondiscriminatory licensing (FRAND).
The patents involved were considered standard and essential, and the Obama administration was concerned the import ban would give Samsung “undue leverage.”
Plus, it only impacted the iPhone 4 and older models of the iPad. Newer iPhones didn’t use an infringing chipset. In the case of the Apple Watch, Apple doesn’t
have any of these arguments.

“In that case 10 years ago, Apple really rolled the dice and the pulled the rabbit out of the hat. It’s very unlikely for them to do that again,” says Levi.

Where does Apple go from here?

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, iPad, Why the Apple Watch is being banned — and how Apple can avoid it, Barack Obama, veto of an import ban on the iPhone, Samsung, Cellular
data patents, Levi, Fair, reasonable, and nondiscriminatory licensing (FRAND), Obama administration, Apple Watch, iPhone, The Verge
</item 10>
=== END ITEM id=39c4f5cf6ea43e82 ===

=== ITEM id=299d1aa8e09b8445 ===
Gold sentences and facts:
Sentence 0: New customers can get up to $1000 in bonus bets if they lose their first bet.
  Fact 0.0: New customers can get up to $1000 in bonus bets.
  Fact 0.1: New customers get bonus bets if they lose their first bet.
Sentence 1: We expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses.
  Fact 1.0: The writer expects Caesars to offer the same offer in VT as it has elsewhere.
  Fact 1.1: New sign-ups would be eligible to receive up to a $1,000 Bonus Bet if their first bet loses.

Retrieved items:
<item 1>
Graph query row:
article: Line Shopping in Sports Betting - How to Find the Best Odds
published_on: 2023-11-06T22:21:05+00:00
publisher: Sporting News
publisher_connection: PUBLISHED_BY
brand: Caesars Sportsbook
brand_connection: MENTIONS
</item 1>

<item 2>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.

Claim here: Bet365 bonus code TSNCO for promo of your choice

FanDuel promo code for Bet any $5 moneyline, Get $150 in bonus bets if your team wins in CBK game

FanDuel has a new offer when using our exclusive FanDuel promo code link. Bet $5 on any moneyline, and Get $150 in bonus bets if your team wins!

It's important to note that there are no odds restrictions, so it'd be prudent to bet a big favorite over an outmatched underdog on the CBK schedule. You need
to win the wager to receive $150 in bonus bets.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> BetMGM
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> BetMGM
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> BetMGM
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -MENTIONS-> BetMGM
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> BetMGM

Entities: BetMGM, Virginia, college basketball, Bet365, Caesars Sportsbook, Sporting News, DraftKings, FanDuel, BetRivers, Best sports betting promo codes for
college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers, New Jersey
</item 2>

<item 3>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking our Caesars Sportsbook promo code link and sign up for a new account. If your first bet loses, Caesars will credit your account with bonus
bets in the amount of your wager up to $1,000.

Click to claim: Caesars Sportsbook promo code NEWS1000 for First Bet up to $1K on Caesars

BetMGM bonus code TSN50 unlocks $1,550 in bonus bets for Blue Devils vs. Razorbacks

Use our exclusive BetMGM bonus code TSN50 to claim a First Bet bonus of up to $1,500, and a $50 bonus bet for customers in select states.

Click our BetMGM bonus code link and register for your new account with promo code TSN50 to unlock this bonus.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Bet365
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> NEWS1000
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Bet365
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News

Entities: Bet365, NEWS1000, Sporting News, BetRivers, DraftKings, TSN50, Blue Devils, Razorbacks, Blue Devils vs. Razorbacks, Best sports betting promo codes
for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers, Caesars Sportsbook, BetMGM, FanDuel
</item 3>

<item 4>
Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars Sportsbook,
DraftKings and FanDuel  (Sporting News, 2023-12-18T21:00:03+00:00)
To claim this top-notch welcome offer, click our Caesars Sportsbook promo code link and sign up for a new Caesars account. Be sure Caesars Sportsbook promo code
NEWS1000 is entered when prompted. Then make your first deposit ($10 minimum).

After that, place your first Caesars Sportsbook bet and if it loses you’ll get the amount of that wager (up to $1,000) as a single bonus bet.

Get up to $1,200 in bonuses for Eagles vs.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, BetMGM, Eagles, Bet365, Caesars Sportsbook, NFL Monday Night Football: Eagles vs. Seahawks, sportsbook bonus offers, Caesars Sportsbook promo
code NEWS1000, Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers,
Caesars Sportsbook, DraftKings and FanDuel, Sporting News, Seahawks, DraftKings, FanDuel, BetRivers
</item 4>

<item 5>
2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet (CBSSports.com, 2023-10-13T21:17:17+00:00)
4. BetMGM Sportsbook: The signup offer at BetMGM caters more to larger players, a $1500 risk free offer which will credit a customer with bonus bets in the
event their first real money wager loses. The boosts and risk free offers on the site cater to smaller players as the limits for those tokens are generally less
than the other sites. With a friendly interface and a rewards program that can be redeemed for bonus bets or MGM Resort credit, this sportsbook is one of the
best in the business for a reason.

5. Caesars Sportsbook: Similar to MGM, the rewards program is what Caesars is most known for in that customers can redeem their wagering credits for use at
Caesars properties. New customers can get up to $1000 in bonus bets if they lose their first bet. The sportsbook offers an extensive betting menu and
specializes in specific boosts related to your home state teams.

6.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Lamar Jackson
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> NFL
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> MLB
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> NLCS
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> markets

Entities: 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet, MGM, MGM Resort, Caesars properties,
CBSSports.com, BetMGM Sportsbook, Kentucky, Caesars Sportsbook, sports betting
</item 5>

<item 6>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
This offer requires a minimum $10 deposit, but you can deposit up to the amount of your first bet to $1,500 to maximize your potential winnings. If your first
bet on the app loses, BetMGM will refund your account with bonus bets in the amount of the original wager up to $1,500.

The extra $50 bonus bet offer is not available in Washington D.C., Kentucky, Mississippi, Nevada, New York, Puerto Rico and Ontario. Customers in those areas
can use BetMGM bonus code SPORTSPICK for the $1,500 First Bet offer.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> BetMGM
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> BetMGM
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> BetMGM
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -MENTIONS-> BetMGM
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> BetMGM

Entities: BetMGM, SPORTSPICK, Ontario, college basketball, New York, Nevada, Bet365, Caesars Sportsbook, Sporting News, DraftKings, FanDuel, BetRivers, Puerto
Rico, Kentucky, Washington D.C.
</item 6>

<item 7>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
DraftKings | BetMGM vs. FanDuel | Caesars vs. BetMGM

Graph facts:
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Bet365
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Bet365
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> college basketball
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> college basketball
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Bet365, college basketball, Sporting News, BetRivers, DraftKings, sports betting promo codes, Best sports betting promo codes for college basketball:
Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers, Caesars Sportsbook, BetMGM, FanDuel
</item 7>

<item 8>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
7 Tennessee vs. No. 14 North Carolina. Duke vs. Arkansas is another marquee matchup on tonight's slate of CBK games.

Claim thousands in sports betting bonuses for college basketball games with these promo codes and links

Top-rated sportsbooks -- BetMGM, Bet365, FanDuel, Caesars Sportsbook, DraftKings and BetRivers -- make it worth your while when signing up as a new user to the
tune of $5K+ in bonus bets!

Click our links to activate the best sports betting bonuses in the industry and register your sportsbook accounts now!

Caesars Sportsbook promo code NEWS1000 for $1,000 First Bet on Caesars for Volunteers vs. Tar Heels

Caesars Sportsbook promo code NEWS1000 gives new users a First Bet up to $1,000 on Caesars!

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Bet365
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> college basketball
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> NEWS1000
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Bet365
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> college basketball

Entities: Bet365, NEWS1000, college basketball, Sporting News, Tennessee, BetRivers, DraftKings, Duke, Arkansas, Tennessee vs. No. 14 North Carolina, Duke vs.
Arkansas, Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings,
BetRivers, North Carolina, Caesars Sportsbook, BetMGM
</item 8>

<item 9>
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

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Caesars Rewards
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont

Entities: Caesars Rewards, Vermont, Bonus Bet, NEWS1000, BetRivers Sportsbook, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting News, Caesars
Sportsbook, sports betting
</item 9>

<item 10>
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

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> BetMGM
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> BetMGM
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> BetMGM
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -MENTIONS-> BetMGM
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> BetMGM

Entities: BetMGM, SPORTSRIV, college basketball, Bet365, Caesars Sportsbook, Sporting News, DraftKings, FanDuel, BetRivers, Duke, Arkansas, Duke vs. Arkansas,
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
</item 10>
=== END ITEM id=299d1aa8e09b8445 ===

=== ITEM id=82b18a4630ca1243 ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: She said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: He has been out of work since his Blues tenure came to a close back in April.
  Fact 1.1: It would be a hard sell to fans.
  Fact 1.2: His time with the London club went so poorly.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Sporting director Matt Crocker made the final decision to hire Hayes after a search process that began in August, following the team’s surprising exit in the
round of 16 at the World Cup and the subsequent departure of head coach Vlatko Andonovski.

“She has tremendous energy and an insatiable will to win,” U.S. Soccer president Cindy Parlow Cone said in a statement. “Her experience in the USA, her
understanding of our soccer landscape and her appreciation of what it means to coach this team makes her a natural fit for this role and we could not be more
pleased to have her leading our women’s national team forward.”

Though Hayes is seen as one of the world’s top coaches in women’s soccer, the appointment still comes as something of a surprise. Here’s how the deal got done.

Details of the deal

At least part of the surprise surrounding Hayes’ hire – and the six-month runway before she officially takes charge – is down to U.S. Soccer’s own messaging.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, World Cup, U.S. Women’s Soccer Team, Matt Crocker, Vlatko Andonovski, Cindy Parlow Cone, women’s soccer, U.S. Soccer, Emma Hayes,
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea, USA
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Chicago, U.S. Women’s Soccer Team, U.S. Soccer, international window, Georgia, Emma Hayes, How U.S. Women’s Soccer Team Lured Emma
Hayes Away From Chelsea, Chelsea, Atlanta, Champions League final, London
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, club football, U.S. Women’s Soccer Team, U.S. Soccer, Georgia, WSL, Chelsea announcement of Hayes’ departure, Emma Hayes, How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea, the U.S., Harry, soccer
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Getty Images, U.S. Women’s Soccer Team, Emma Hayes, Millie Bright, Denise Reddy, United States Under-20, Chicago Red Stars, Paul
Green, Marc Atkins, How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea, England, New Jersey, football
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, Chelsea Women, U.S. women’s national team, Women’s Super League, club football, U.S. Soccer, Emma Hayes, How U.S.
Women’s Soccer Team Lured Emma Hayes Away From Chelsea
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, U.S. Women’s Soccer Team, U.S. Soccer, Andonovski, Emma Hayes, How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea,
Chelsea, Berhalter
</item 6>

<item 7>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, English Football League, Emma Hayes, U.S. Women's Soccer Team, Spain's national team, Spain's federation, Chelsea's away game at
Aston Villa, How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea, Aston Villa, Olympics
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Crocker, U.S. Soccer, U.S. Women's National Soccer Team, Emma Hayes, How U.S. Women’s Soccer Team Lured Emma Hayes Away From
Chelsea, Chelsea
</item 8>

<item 9>
Graph query row:
article_title: How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea
publisher: The New York Times
person: Emma Hayes
organization: Chelsea Women
passage: How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, U.S. Women’s Soccer Team, Emma Hayes, Paul Green, How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea, Chelsea
</item 10>
=== END ITEM id=82b18a4630ca1243 ===
