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

=== ITEM id=fa5d9087661f3176 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes with Apple.
  Fact 0.1: The Justice Department is focused on the deals Google makes with Samsung.
  Fact 0.2: The Justice Department is focused on the deals Google makes with Mozilla.
  Fact 0.3: The Justice Department is focused on the deals Google makes with many others.
  Fact 0.4: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: The last roundup reported that Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.

Graph facts:
- Meagan Bellshaw -WORKS_FOR-> U.S. Justice Department
- Cue -SAID-> search engines
- Cue -PARTICIPATED_IN-> Cue’s testimony
- 5 things we learned so far about the Google antitrust case -MENTIONS-> Meagan Bellshaw
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue

Entities: Cue, Meagan Bellshaw, Apple's own search product, Google-Apple revenue-share deal, Apple defends Google Search deal in court: ‘There wasn’t a valid
alternative’, Google Search, Apple, Pichai, Google, Justice Department, The Verge
</item 1>

<item 2>
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
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> US v. Google

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, ISA, privacy pop-ups, search engines, Google Search, DuckDuckGo,
Apple, Google, trial, The Verge
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.

Graph facts:
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> Apple's own search product
- Cue -SAID-> search engines
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: US Department of Justice, Information Services Agreement, Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Sundar
Pichai, Google Search, Apple, Google, privacy, The Verge
</item 6>

<item 7>
Graph query row:
entities_and_articles: [{'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Google Nest', 'published_on': None}, {'labels': ['__KGBuilder__',
'Article', '__Entity__'], 'name': "200+ of the best deals from Amazon's Cyber Monday sale", 'published_on': '2023-11-27T08:45:59+00:00'}, {'labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Apple deals', 'published_on': None}]
relationships: ['MENTIONS', 'MENTIONS']
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, search engine market, Google Search, Apple, Google, Justice
Department, The Verge
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, ISA, Google Search deal, Google’s privacy policies,
Apple’s financial filings, Google Search, Apple, DOJ, Google, user privacy, Safari, The Verge
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=fa5d9087661f3176 ===

=== ITEM id=befa85d0b17be160 ===
Gold sentences and facts:
Sentence 0: The Buckeyes settled for a 52-yard field goal late in the first half, which Jayden Fielding missed wide left.
  Fact 0.0: The Buckeyes settled for a 52-yard field goal late in the first half.
  Fact 0.1: Jayden Fielding missed the 52-yard field goal wide left.
Sentence 1: His 31-yard field goal is good, and the Buffs take the first lead of the game on their first drive.
  Fact 1.0: His 31-yard field goal is good.
  Fact 1.1: The Buffs take the first lead of the game on their first drive.

Retrieved items:
<item 1>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:42 p.m.: A 15-yard facemask penalty on Broncos WR Marvin Mims Jr. backed up the offense and the Vikings' defense held strong again as a result. Here comes
Will Lutz for his fifth field goal attempt of the game.

10:38 p.m.: INTERCEPTION! Joshua Dobbs was hit as he threw and Broncos CB Ja'Quan McMillan came up with the pick. It was LB Josey Jewell who brought the
pressure on Dobbs. Denver takes over from the Vikings' 9-yard line after McMillan picked up nine yards on the interception return.

Vikings 17, Broncos 12

10:34 p.m.: FIELD GOAL — Will Lutz buries his fourth field goal of the day, this time from 37 yards. Lutz has been automatic in this one, going a perfect
4-for-4 to account for all of Denver's points.

10:32 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Ja'Quan McMillan, Josey Jewell, Vikings, Joshua Dobbs, Will Lutz, Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night
Football' game, Sporting News, Marvin Mims Jr., NFL 'Sunday Night Football' game, Broncos
</item 1>

<item 2>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
PENALTY – Sun Devils Ed Woods is ejected for targeting. He will miss the remainder of this game as well as the first half against Washington in two weeks.

9:10 p.m. – That's the fourth sack allowed by the Colorado O-line. The Buffaloes now lead FBS with most sacks allowed this season at 30.

8:52 p.m. – Skattebo is heating up under the sun. He's added 24 rushing yards from Arizona State's possession.

End of second quarter: Colorado 14, Arizona Sate 17

Colorado 14, Arizona State 17

8:20 p.m. FIELD GOAL – Dario Longhetto nails a 33-yard field goal to end the half. The Sun Devils have a narrow lead over the Buffs at the halfway mark.

Colorado 14, Arizona State 14

8:14 p.m. TOUCHDOWN – Sanders ties it up! The QB keeps it for 16 yards and a touchdown.

Graph facts:
- Colorado -MENTIONS-> bowl eligibility
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> Colorado
- College football odds, lines, schedule for Week 6: Texas, Alabama, Notre Dame among favorites in ranked games -MENTIONS-> Colorado
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Colorado
- Shilo Sanders -WORKS_FOR-> Colorado

Entities: Colorado, Sanders, targeting, Sporting News, Washington, Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game,
Colorado vs. Arizona State college football game, Ed Woods, Skattebo, Dario Longhetto, Arizona State
</item 2>

<item 3>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado takes over on the ASU 25.

SHEDEUR TO ANTONIO FOR A HUGE GAIN 🦬@CUBuffsFootball pic.twitter.com/tvfWjRWWAE — FOX College Football (@CFBONFOX) October 8, 2023

Colorado 24, Arizona State 24

10:11 p.m. TOUCHDOWN – Don't count the Sun Devils out yet! Bourguet finds Troy Omeire on a 15-yard pass to tie it up. Only 50 seconds remain.

9:48 p.m. – It's blocked! Carter Stoutmire gets in front of the 52-yard field goal attempt from Longhetto to deny a Sun Devil score.

9:38 p.m. FIELD GOAL – Alejandro Mata nails a 42-yard field goal to extend Colorado's lead to seven. Ten minutes are left in the game.

9:35 p.m. – That's a huge run for the Buffaloes!

Graph facts:
- Colorado -MENTIONS-> bowl eligibility
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> Colorado
- College football odds, lines, schedule for Week 6: Texas, Alabama, Notre Dame among favorites in ranked games -MENTIONS-> Colorado
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Colorado
- Shilo Sanders -WORKS_FOR-> Colorado

Entities: Colorado, Alejandro Mata, Sporting News, Antonio, Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game,
Colorado vs. Arizona State college football game, Shedeur, Bourguet, Troy Omeire, Carter Stoutmire, Longhetto, FOX College Football, Arizona State
</item 3>

<item 4>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado fans, you're back. For now.

A 43-yard pass from Shedeur Sanders on the Buffaloes' final drive helped set up Alejandro Mata for the game-winning 43-yard field goal with 12 seconds left,
delivering Colorado the 27-24 win over Arizona State for their first Pac-12 win of the season. It snaps their eight-game conference losing streak.

Sanders had a mediocre day by his lofty standards, but his 239 yards and a touchdown were enough in perhaps Colorado's most crucial game so far. It brought the
team to 4-2, but had it lost, it would have been much harder for the Buffs to finish the season bowl eligible. A win gives them a leg up in a competitive
conference, particularly facing four ranked teams down the stretch of the season.

If Colorado wants to solidify a bowl game, it'll have to fix many things, one of them being the offensive line.
</item 4>

<item 5>
Colorado vs. Arizona State live score, updates, highlights from Week 6 college football game (Sporting News, 2023-10-07T21:21:03+00:00)
Colorado needed a bounce back win and got one. Meanwhile Arizona State kept on struggling, falling to 1-5.

MORE: Watch Colorado vs. Arizona State live with Fubo (free trial)

The Sporting News tracked updates and highlights from college football's Week 6 game between Colorado and Arizona State.

Colorado vs. Arizona State score

Q1 Q2 Q3 Q4 Final Colorado 7 7 0 13 27 Arizona State 7 10 0 7 24

Colorado vs. Arizona State results, highlights from Week 6 game

FINAL: Colorado 27, Arizona State 24

Colorado 27, Arizona State 24

10:17 p.m. FIELD GOAL – Absolutely clutch. Mata makes a 43-yard field goal look easy to give Colorado the lead with 12 seconds on the clock.

10:14 p.m. – Bullseye! Sanders finds Antonio on a 43-yard pass. Colorado takes over on the ASU 25.

Graph facts:
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Fubo
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Fubo
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> Fubo
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> Fubo
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Fubo

Entities: Fubo, college football, Colorado, Sanders, Mata, Sporting News, Colorado vs. Arizona State Week 6 college football game, Antonio, Colorado vs. Arizona
State live score, updates, highlights from Week 6 college football game, Arizona State
</item 5>

<item 6>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
– Colorado settles for a field goal again, this time for a 39-yarder. It leads UCLA 6-0 with 7:44 remaining in the first quarter.

7:59 p.m. INTERCEPTION – Welcome to Hollywood, Travis Hunter! The two-way star picks off Ethan Garbers at the Bruins' 36-yard line. After review, the
interception stands, and it's Buffalo ball. Colorado defense with a statement play early on.

WHAT A PICK BY TRAVIS HUNTER 🔥@CUBuffsFootball pic.twitter.com/3i9twSG1uq — FOX College Football (@CFBONFOX) October 29, 2023

Colorado 3, UCLA 0

7:54 p.m. FIELD GOAL – Sanders can't find the end zone, but Alejandro Mata finds space between the uprights. His 31-yard field goal is good, and the Buffs take
the first lead of the game on their first drive.

7:52 p.m.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> college football
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> college football
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> college football
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> college football
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -MENTIONS-> college football

Entities: college football, Colorado vs. UCLA live score, updates, highlights from Week 9 college football game, UCLA, Ethan Garbers, Colorado, Sanders, Travis
Hunter, FOX College Football, Colorado vs. UCLA college football game, Travis Hunter interception of Ethan Garbers, Alejandro Mata, 31-yard field goal,
39-yarder field goal, Sporting News
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Who do you want lining up the potential winning kick in a Super Bowl? It’s an interesting question in a season that has seen Moody and Saints rookie Blake Grupe
miss game-winning tries, but the data could be on the 49ers’ side.

FGA Dist Rookies Veterans <33 Yds 97% [31-32] 94% [330-351] 34-44 Yds 75% [21-28] 82% [230-279] 45-50 Yds 78% [7-9] 73% [121-165] 51-55 Yds 100% [8-8] 59%
[33-56] 56+ Yds 0% [0-1] 55% [6-11] Totals 86% [67-78] 84% [720-862]

The table above shows field-goal make rates in the postseason since 2000 for rookie kickers vs. veterans.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Saints, Super Bowl, Blake Grupe, field-goal make rates, postseason since 2000, N.F.L. Executives on the Biggest Questions Facing
Super Bowl Contenders, 49ers, Moody
</item 7>

<item 8>
Colorado vs. UCLA live score, updates, highlights from Week 9 college football game (Sporting News, 2023-10-28T22:08:03+00:00)
Can the Buffaloes finally connect for a touchdown before its too late?

UCLA adds on another 💪@UCLAFootball makes it 21-9 in the 4th quarter vs Colorado pic.twitter.com/6Eu9gn6kgn — FOX College Football (@CFBONFOX) October 29, 2023

End of third quarter: Colorado 9, UCLA 14

10:20 p.m. – What a catch for the Bruin! After review, Garbers pass to Loya is ruled complete.

Colorado 9, UCLA 14

10:06 p.m. FIELD GOAL – Mata nails a 34-yard field goal to lessen Colorado's deficit to five. Four minutes remain in the third quarter.

10:03 p.m. – Sacked! Sanders is taken down by Laiatu Latu, who has multiple tonight. UCLA records its sixth sack of the night with the latest.

Graph facts:
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> Fubo
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> ESPN
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> college football
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> Deion Sanders
- Colorado vs. UCLA live score, updates, highlights from Week 9 college football game -MENTIONS-> Week 9 college football game

Entities: Colorado vs. UCLA live score, updates, highlights from Week 9 college football game, UCLA, Colorado, Sanders, FOX College Football, Colorado vs. UCLA
college football game, Garbers, Loya, Mata, Laiatu Latu, Sporting News
</item 8>

<item 9>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
COURTLAND SUTTON INCREDIBLE 4TH DOWN GRAB ‼️



🎥: @NFL

pic.twitter.com/T0Mrch9Tf9 — The Sporting News (@sportingnews) November 20, 2023

Vikings 20, Broncos 15

11:02 p.m.: FIELD GOAL — Greg Joseph knocks it in from 30 yards. The Vikings' offense was moving, and it did its job by chewing a lot of clock on a 14-play,
seven-minute and 20-second drive. Minnesota stalled out on Denver's 12-yard line, settling for a field goal.

Can the Vikings' defense keep the Broncos out of the end zone as it has all night? The drive of the game is coming up.

10:57 p.m.: The spot on the field was overturned! Ty Chandler was ruled just short of the line to gain, setting up a 4th & inches.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Greg Joseph, Ty Chandler, Vikings, Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game, Minnesota,
Courtland Sutton, Sporting News, NFL 'Sunday Night Football' game, Denver, Broncos
</item 9>

<item 10>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Vikings decided to go for it and Joshua Dobbs picked up the first down on a QB sneak.

10:54 p.m.: Ty Chandler continues to rumble. The spell RB picked up a massive gain on 3rd & 20 and it appeared that he got the first down, but the spot on the
field is under review. This could be a crucial call.

10:48 p.m.: It looked like it was going to be a three-and-out for the Vikings offense but a fake punt has kept the drive alive! Ty Chandler picked up 31 yards
on a gutsy call from head coach Kevin O'Connell to bring the ball into Denver's territory.

Vikings 17, Broncos 15

10:43 p.m.: FIELD GOAL — Will Lutz knocks one in from 28 yards, his fifth made field goal of the day. Lutz has been the Broncos' lone source of offense,
bringing Denver within one score with 10:37 to play in the game.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Ty Chandler, Vikings, Joshua Dobbs, Vikings vs. Broncos NFL 'Sunday Night Football' game, Will Lutz, Vikings vs. Broncos live score, updates,
highlights from NFL 'Sunday Night Football' game, Kevin O'Connell, Sporting News, Broncos
</item 10>
=== END ITEM id=befa85d0b17be160 ===

=== ITEM id=b23a145289b79273 ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: Huang testified during cross-examination.
  Fact 0.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.2: Paradigm was told that Bankman-Fried didn't want investors as directors.
  Fact 0.3: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: The criminal trial of Sam Bankman-Fried was highly anticipated.
  Fact 1.1: Sam Bankman-Fried is the former CEO of FTX.
  Fact 1.2: FTX is a bankrupt crypto exchange.
  Fact 1.3: The criminal trial of Sam Bankman-Fried started on Tuesday.
  Fact 1.4: The trial is to determine whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Damian Williams
- Damian Williams -SAID-> financial fraud
- Damian Williams -PARTICIPATED_IN-> Press conference
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto

Entities: Damian Williams, crypto, alleged crimes, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand, Fortune, FTX, the government, Sam Bankman-Fried, the trial, MIT, fraud
</item 1>

<item 2>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried trial, wire fraud, Ellison, FTX, Sam Bankman-Fried,
conspiracy to take FTX customer funds, fake balance sheets, How is it still getting worse for Sam Bankman-Fried?, Genesis, The Verge
</item 2>

<item 3>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> SBF Trial
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google’s hardware event
- Founders, are events useful? -MENTIONS-> Pixel 8
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Pixel 8
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Pixel 8 Pro

Entities: teens’ startup, ad-free tier, Pixel 8, Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier, Jacquelyn,
Southern District of New York courthouse, Bankman-Fried, SBF and FTX trial, TechCrunch, TechCrunch+, the defense, FTX, Google, Rebecca, the prosecution
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Damian Williams
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The best new TV series on every streaming service -MENTIONS-> Ponzi scheme
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison

Entities: Ponzi scheme, crypto markets, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda, Ellison, Cohen, Fortune, FTX, Sam Bankman-Fried
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda Research, FTX, Nishad Singh, Wang, Sam Bankman-Fried, The Verge
</item 6>

<item 7>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Barbara Fried, the defense, FTX, the government, Sam Bankman-Fried, How is it still getting worse for Sam Bankman-Fried?, the recording,
Drappi, Joseph Bankman, FTX’s philanthropic arm, Zac Prince, BlockFi, the trial, FTX’s bankruptcy
</item 7>

<item 8>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, SBF’s trial, FTT tokens, BUSD, Chapter 11 bankruptcy, bank run on FTX, John J. Ray III, DealBook Summit, SBF’s trial starts soon,
but how did he — and FTX — get here?, Alameda, TechCrunch, Bahamas, Enron, FTX, Sam Bankman-Fried
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> crypto
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto

Entities: crypto, Alameda, FTX trial, second trial, misappropriated FTX customer deposits, political contributions, The FTX trial is bigger than Sam
Bankman-Fried, real estate, FTX, Sam Bankman-Fried, The Verge
</item 9>

<item 10>
Graph query row:
report: SBF’s trial starts soon, but how did he — and FTX — get here?
published_on: 2023-10-01T14:00:29+00:00
publishers: ['TechCrunch']
source_passages: [{'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)', 'passage': 'SBF’s trial
starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nWithin two years, more than 80 investors provided about $2 billion in
capital to FTX, helping Bankman-Fried propel his vision into a reality. In January 2022, the company raised $400 million in a Series C round, boosting its
valuation to $32 billion. That was its last round of public funding.\n\nThe company gained somewhat mainstream recognition with branding deals and partnerships.
For example, in 2021 it bought the naming rights for the Miami Heat’s home arena. FTX also got its name branded on Major League Baseball umpires’ polos, and it
partnered with celebrities like Tom Brady and his ex-wife, Gisele Bündchen, as well as Steph Curry, Shaquille O’Neal and Naomi Osaka, among others. He also had
close ties to U.S. regulators and government officials, many of whom he donated to.'}, {'page': 'SBF’s trial starts soon, but how did he — and FTX — get here?
(TechCrunch, 2023-10-01T14:00:29+00:00)', 'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch,
2023-10-01T14:00:29+00:00)\nregulators and government officials, many of whom he donated to.\n\nBankman-Fried was even compared to Warren Buffet and many called
him the white horse of crypto (TechCrunch never did, for what it’s worth).\n\nBut in early November 2022, that all changed.\n\nFTX’s collapse\n\nConcerns
surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8 billion in
liabilities as of June 30, 2022.\n\nBut there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of
“FTT collateral.” FTT was the token behind FTX.\n\nThe balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the
total FTT market cap, which was about $3 billion at the time.'}, {'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch,
2023-10-01T14:00:29+00:00)', 'passage': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nThat means it
purported to have more FTT tokens on its balance sheet than what existed in the world.\n\nAround the same time it was exposed, the world’s largest crypto
exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and FTT. (It had an equity position in FTX from 2019 to 2021.) This
essentially triggered a bank run on FTX.\n\nFTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J.
Ray III, the Enron turnaround veteran, was appointed its new CEO.\n\nBankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit,
he appeared virtually from the Bahamas, saying “I didn’t ever try to commit fraud on anyone; I was shocked by what happened this month.”'}, {'page': 'SBF’s
trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)', 'passage': 'SBF’s trial starts soon, but how did he — and FTX
— get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nU.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in
American history,” in a press conference.\n\nThe aftermath\n\nFTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both
pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and
Exchange Commission (SEC) and Commodity Futures Trading Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors
and will be major witnesses in the trial, given their close ties to Bankman-Fried, FTX and Alameda.\n\nIn January, Bankman-Fried pleaded not guilty to all
counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer funds. He could face up to 115 years in jail if
convicted on all charges.'}, {'page': 'SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)', 'passage': 'SBF’s
trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)\nHe could face up to 115 years in jail if convicted on all
charges.\n\nThe crypto industry as a whole suffered from FTX’s collapse, which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did
Genesis Global Trading in January.\n\nWhere we are today\n\nBankman-Fried will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense
attorney and former federal prosecutor, will be the lead attorney. If that name sounds familiar, it might be because he also represented Ghislaine Maxwell in
her sex trafficking trial related to Jeffrey Epstein. He requested an early release for Bankman-Fried but was denied.\n\nWith the trial beginning, we’re
witnessing how FTX’s story ends. But what’s on our mind is what happens to the investors and creditors affected by the collapse? And what happens to the
billions in crypto assets tied up in legal proceedings?'}]
connections: [{'entity': 'The New York Times', 'relationship': 'MENTIONS'}, {'entity': 'FTX collapse', 'relationship': 'MENTIONS'}, {'entity': 'SBF',
'relationship': 'MENTIONS'}, {'entity': 'SBF’s trial', 'relationship': 'MENTIONS'}, {'entity': 'Series C round', 'relationship': 'MENTIONS'}, {'entity': 'Gisele
Bündchen', 'relationship': 'MENTIONS'}, {'entity': 'Shaquille O’Neal', 'relationship': 'MENTIONS'}, {'entity': 'Naomi Osaka', 'relationship': 'MENTIONS'},
{'entity': 'Warren Buffet', 'relationship': 'MENTIONS'}, {'entity': 'CoinDesk', 'relationship': 'MENTIONS'}, {'entity': 'FTX’s collapse', 'relationship':
'MENTIONS'}, {'entity': 'liquidity', 'relationship': 'MENTIONS'}, {'entity': 'FTT tokens', 'relationship': 'MENTIONS'}, {'entity': 'BUSD', 'relationship':
'MENTIONS'}, {'entity': 'Chapter 11 bankruptcy', 'relationship': 'MENTIONS'}, {'entity': 'bank run on FTX', 'relationship': 'MENTIONS'}, {'entity': 'John J. Ray
III', 'relationship': 'MENTIONS'}, {'entity': 'DealBook Summit', 'relationship': 'MENTIONS'}, {'entity': 'Damian Williams', 'relationship': 'MENTIONS'},
{'entity': 'Gary Wang', 'relationship': 'MENTIONS'}, {'entity': 'Caroline Ellison', 'relationship': 'MENTIONS'}, {'entity': 'Bankman-Fried’s trial',
'relationship': 'MENTIONS'}, {'entity': 'Wang and Ellison pleaded guilty to federal criminal charges', 'relationship': 'MENTIONS'}, {'entity': 'Bankman-Fried
pleaded not guilty', 'relationship': 'MENTIONS'}, {'entity': 'financial fraud', 'relationship': 'MENTIONS'}, {'entity': 'conspiracy to commit money laundering',
'relationship': 'MENTIONS'}, {'entity': 'conspiracy to misuse customer funds', 'relationship': 'MENTIONS'}, {'entity': 'civil penalties', 'relationship':
'MENTIONS'}, {'entity': 'federal criminal charges', 'relationship': 'MENTIONS'}, {'entity': 'BlockFi Chapter 11 filing', 'relationship': 'MENTIONS'}, {'entity':
'Genesis Global Trading', 'relationship': 'MENTIONS'}, {'entity': 'Genesis Global Trading Chapter 11 filing', 'relationship': 'MENTIONS'}, {'entity': 'Cohen &
Gresser', 'relationship': 'MENTIONS'}, {'entity': 'Ghislaine Maxwell', 'relationship': 'MENTIONS'}, {'entity': 'sex trafficking trial', 'relationship':
'MENTIONS'}, {'entity': 'all charges', 'relationship': 'MENTIONS'}, {'entity': 'crypto assets', 'relationship': 'MENTIONS'}, {'entity': 'legal proceedings',
'relationship': 'MENTIONS'}, {'entity': 'Bankman-Fried', 'relationship': 'MENTIONS'}, {'entity': 'crypto', 'relationship': 'MENTIONS'}, {'entity': 'Mark Cohen',
'relationship': 'MENTIONS'}, {'entity': 'Alameda Research', 'relationship': 'MENTIONS'}, {'entity': 'Alameda', 'relationship': 'MENTIONS'}, {'entity': 'Miami
Heat', 'relationship': 'MENTIONS'}, {'entity': 'FTT', 'relationship': 'MENTIONS'}, {'entity': 'government officials', 'relationship': 'MENTIONS'}, {'entity':
'wire fraud', 'relationship': 'MENTIONS'}, {'entity': 'Bahamas', 'relationship': 'MENTIONS'}, {'entity': 'Enron', 'relationship': 'MENTIONS'}, {'entity': 'FTX',
'relationship': 'MENTIONS'}, {'entity': 'Major League Baseball', 'relationship': 'MENTIONS'}, {'entity': 'Press conference', 'relationship': 'MENTIONS'},
{'entity': 'Sam Bankman-Fried', 'relationship': 'MENTIONS'}, {'entity': 'BlockFi', 'relationship': 'MENTIONS'}, {'entity': 'Tom Brady', 'relationship':
'MENTIONS'}, {'entity': 'Binance', 'relationship': 'MENTIONS'}, {'entity': 'Commodity Futures Trading Commission', 'relationship': 'MENTIONS'}, {'entity': 'U.S.
Securities and Exchange Commission', 'relationship': 'MENTIONS'}, {'entity': 'crypto industry', 'relationship': 'MENTIONS'}, {'entity': 'U.S. regulators',
'relationship': 'MENTIONS'}, {'entity': 'Jeffrey Epstein', 'relationship': 'MENTIONS'}, {'entity': 'U.S.', 'relationship': 'MENTIONS'}, {'entity': 'fraud',
'relationship': 'MENTIONS'}, {'entity': 'Steph Curry', 'relationship': 'MENTIONS'}]
</item 10>
=== END ITEM id=b23a145289b79273 ===
