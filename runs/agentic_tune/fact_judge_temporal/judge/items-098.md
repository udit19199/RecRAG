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

=== ITEM id=c7edde143d7e9a05 ===
Gold sentences and facts:
Sentence 0: When he entered free agency, a dozen teams lined up, curious to see if they could meet his eye.
  Fact 0.0: A dozen teams lined up when the free agent entered free agency.
  Fact 0.1: The dozen teams were curious to see if they could meet the free agent's eye.
Sentence 1: As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and determining the fate of those around me.
  Fact 1.0: The player explores a system of the Imperium.
  Fact 1.1: The player explores with a loyal crew of companions.
  Fact 1.2: The player makes impactful choices.
  Fact 1.3: The player determines the fate of those around the player.

Retrieved items:
<item 1>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I create my custom protagonist, determining stats and bonuses based on my backstory, home world, and so on. I’m then thrust into a high-pressure job interview
as a potential heir to a Rogue Trader aboard her flagship. Little do we know that there’s a coup in the works, and traitors aboard the ship. After a deadly
struggle against heretics and demons, I ascend to the position of Rogue Trader. I’m the captain now, and I get to decide how to run my ship.

This is a vast game, with tons of features you’d expect from a CRPG — companions (each with their own narrative paths and conversation trees), top-down
strategic combat, and branching choices that impact the world around you. I get to make choices all day — in fact, that’s part of the gig of a Rogue Trader.
Some are more important than others; whenever I get to specific points in the story, I can choose from one of three major paths.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Warhammer 40K, CRPG, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K, Rogue Trader
</item 1>

<item 2>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
“In the grim darkness of the far future, there is only war.” That’s the tagline of Warhammer 40,000, one of the most over-the-top and brutal sci-fi settings
around. But even in a merciless, brutal dystopia that grinds its people into dust, some get to enjoy being on the top of the food chain. Warhammer 40,000: Rogue
Trader is about the perils and pleasures of being atop that particular hierarchy.

Rogue Traders in 40K are freelance explorers, scouting the far frontiers of the Imperium of Man. It’s a risky career, but it comes with luxurious rewards:
agency, freedom, power, and a giant flagship. As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and
determining the fate of those around me.

Rogue Trader is a computer role-playing game in the vein of Baldur’s Gate 3 or Pillars of Eternity.

Graph facts:
- The best video games of the year so far -MENTIONS-> Warhammer 40,000: Rogue Trader
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon

Entities: Warhammer 40,000: Rogue Trader, Polygon, Baldur’s Gate 3, Warhammer 40,000, Imperium of Man, Pillars of Eternity, Rogue Traders, Rogue Trader’s
role-playing embraces the brutality and freedom of Warhammer 40K
</item 2>

<item 3>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
Dogmatic choices exult the God-Emperor and loathe the mutant and the witch, the Heretical options pursue corruptive power, and the Ionoclast path is the closest
thing we have to modern-day “good guy” morality.

After the events of the tutorial, my beautiful voidship is run down and my staff is struggling. As the newly anointed Rogue Trader, I have to trek around the
various planets of the Koronus Expanse to get a new Navigator, fix my ship up, and avoid any major diplomatic incidents with the locals. My voidship is the size
of a modern city; I’m as much a governor as a captain, and I have to manage the ship, its cargo, and its many occupants.

In the process of getting back on my feet, I uncover a nefarious cult and a deep conspiracy. It’s not an easy job, but I’m blessed with a handful of companions
from the Imperium to help me out.
</item 3>

<item 4>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
For 40K fans, this is a rare treat — a game that digs past the heroic facade of bolters and battles and taps into the grimdark dystopia that makes this
particular sci-fi setting so damn compelling.

Warhammer 40,000: Rogue Trader will be released on Dec. 7 on Windows PC. The game was reviewed using a pre-release download code provided by Owlcat Games. Vox
Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links.
You can find additional information about Polygon’s ethics policy here .

Graph facts:
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- The best video games of the year so far -MENTIONS-> Windows PC
- The best Xbox controller to buy right now -MENTIONS-> Windows PC
- The best video games of the year so far -MENTIONS-> Warhammer 40,000: Rogue Trader
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Warhammer 40,000: Rogue Trader, Windows PC, Polygon, Vox Media, Warhammer 40K, Rogue Trader’s role-playing embraces the brutality and freedom of
Warhammer 40K, Owlcat Games
</item 4>

<item 5>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Morosi reports Shohei Ohtani is "likely to decide on a team within the next week," according to a source.

Status quo on Yankees' Juan Soto trade talks

Date: Sunday, Dec. 3

Source: Jon Heyman, New York Post

The Yankees reportedly are pushing hard to acquire Juan Soto from the Padres, but Heyman says the teams haven't spoken since San Diego requested a six-player
package in return for Soto and Trent Grisham. That ask included pitchers Michael King and Drew Thorpe, who the Yankees aren't inclined to move. Stay tuned...

Shohei Ohtani field growing smaller

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

It has been expected since before he hit free agency that Ohtani would set the record for the largest fully guaranteed contract in North American sports
history, with Passan reporting the deal could reach as much as $600 million.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Padres
- A.J. Preller -WORKS_FOR-> Padres
- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings -MENTIONS-> Padres
- Top 25 MLB offseason trade candidates -MENTIONS-> Padres
- Yates -WORKS_FOR-> Padres

Entities: Padres, Juan Soto, Yankees, New York Post, Trent Grisham, MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free
agency, Jon Heyman, Morosi, Michael King, Drew Thorpe, ESPN, Sporting News, Shohei Ohtani, Jeff Passan
</item 5>

<item 6>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, MLB, Mariners, Yoshinobu Yamamoto, Ken Rosenthal, Winter meetings, Nashville, Sporting News, Shohei Ohtani, Shohei Ohtani rumors
tracker: Latest news, updates on free agent's MLB future, posting of Yoshinobu Yamamoto for free agency
</item 6>

<item 7>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
All eyes since the start of the offseason have been on Shohei Ohtani. And for the most part, there hasn't been much to see.

The two-way phenom is the most sought-after free agent in MLB history, with his combination of talent and marketability making him a franchise-changing player.

And the bidding for Ohtani could be winding down soon. Multiple reports have indicated Ohtani could sign at some point during the winter meetings, which are
scheduled to take place from Dec. 3 through Dec. 6.

Many of the top teams are expected to be in the mix for Ohtani. The Dodgers and Cubs are among the top teams expected to be aiming to sign the two-time AL MVP,
and plenty of others should continue to compete through the end to land him.

Here's the latest on Ohtani's free agency pursuits.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> MLB
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> MLB
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> MLB
- Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00) -MENTIONS-> MLB
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> MLB

Entities: MLB, Cubs, Winter meetings, Sporting News, Shohei Ohtani, Dodgers, Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free
agency short list emerges, free agency
</item 7>

<item 8>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can call on them to unlock doors or perform other environmental checks, but they come most in handy in combat.

When I meet opposition, it’s deeply satisfying to control my troops in a turn-based battle. Each fight takes place on a grid; it’s very similar to Baldur’s Gate
3 or even XCOM 2. Some positions provide cover, while others are out in the open. Friendly fire is also a very real concern. An arc of auto-fire from a bolter,
or a Navigator’s third eye opening, can harm friend as well as foe. My Rogue Trader is a sniper, and she would be lost without her Senechal taking the front
lines. There’s a lot of firepower at my disposal, and it’s mostly quite satisfying to use — even if I occasionally shred my poor Senechal with a devastating
AOE.

I can chat with the companions between battles, learning more about their pasts.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Baldur’s Gate 3, Warhammer 40K, XCOM 2, Rogue Trader, Senechal, Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer
40K
</item 8>

<item 9>
Graph query row:
lookup: Rogue Trader article
article: Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K
publication_date: 2023-12-06T13:00:00+00:00
publisher: Polygon
relationship: PUBLISHED_BY
connected_entity: Polygon
</item 9>

<item 10>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
'Money's great, but winning's better'

Date: Nov. 21

Source: Max Scherzer, Rangers pitcher on Foul Territory

OK, so this isn't necessarily a rumor, but an example of how much players would love for Ohtani to join their team. Scherzer was a guest on Foul Territory on
Monday.

"Money's great, but winning's better."



Max Scherzer shares his pitch for Shohei Ohtani to join the @Rangers 🏆



▶️ https://t.co/6KoSRqSgk2 pic.twitter.com/PHW2jHWj6C — Foul Territory (@FoulTerritoryTV) November 20, 2023

Maybe Ohtani won’t sign quite so soon after all

Date: Nov.

Graph facts:
- MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 -MENTIONS-> MLB
- Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut -MENTIONS-> MLB
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News

Entities: MLB, Sporting News, Rangers, Shohei Ohtani, Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future, Foul Territory, Max
Scherzer, free agency
</item 10>
=== END ITEM id=c7edde143d7e9a05 ===

=== ITEM id=af9b5840b8152dd0 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Helena World Chronicle.
  Fact 0.1: Helena World Chronicle is an Arkansas-based publisher.
  Fact 0.2: The case argues that Google 'siphons off' news publishers' content.
  Fact 0.3: The case argues that Google 'siphons off' news publishers' readers.
  Fact 0.4: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 0.5: The case argues that this siphoning happens through anticompetitive means.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest that Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- 5 things we learned so far about the Google antitrust case -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, Wilson White, Match,
U.S. attorneys general, Epic's antitrust lawsuit, U.S. District Court in California’s Northern District, Department of Justice, Google's antitrust lawsuit with
the Department of Justice, search monopoly, antitrust, Netflix, Apple
</item 2>

<item 3>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Project Hug
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Project Hug
- Tim Sweeney -SAID-> Project Hug
- Project Hug -PART_OF-> Google Games Velocity Program
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic

Entities: Project Hug, Epic, Google, Epic v. Google, explained, Sweeney, state attorneys general, Project Agave, Project Banyan, game developers, phone makers,
Google's app store, settlement, anticompetitive practices, paying game developers and phone makers not to abandon Google's app store, deleting potentially
incriminating messages
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Android, Galaxy Store, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, OnePlus, Epic Games app, Google
Play Store, Android app developers, Project Hug, Google Games Velocity Program, antitrust battle, agreement with OnePlus, deal between Google and Samsung,
Google developer agreements, anti-competitive practices, Epic Games
</item 4>

<item 5>
Graph query row:
TechCrunch_report: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
TechCrunch_passages: ['News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\nA new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of
anticompetitive behavior in violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based
publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also
specifically cites new AI technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.\n\nIn the complaint,
Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing publishers’
content on Google, losing them “billions of dollars.”\n\nIn addition to new AI technologies, the suit points to Google’s older question-and-answer technologies,
like the “Knowledge Graph” launched in May 2012, as part of the problem.', "News publisher files class action antitrust suit against Google, citing AI’s harms
to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\n“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right
of the search results. This panel contains a summary of content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive
database by extracting information from Publishers’ websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed
databases,'” it says.\n\nBy 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that
Google tapped into was content “misappropriated from Publishers,” the complaint alleges.\n\nOther Google technologies, like “Featured Snippets” where Google
algorithmically extracts answers from webpages, were also cited as shifting traffic away from publishers’ websites.\n\nMore importantly, perhaps, is the suit’s
tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The Wall Street Journal, which led with a
shocking statistic.", 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\nWhen online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the
AI would answer the user’s query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic
going forward, as Google today drives nearly 40% of their traffic, according to data from Similarweb.\n\nSome publishers are now trying to get ahead of the
problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model training. But overall, publishers believe they’ll
lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s report noted.\n\nThe lawsuit reiterates this concern,
claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from visiting the websites of Class members
who are part of the digital news and publishing line of commerce.”', 'News publisher files class action antitrust suit against Google, citing AI’s harms to
their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nSGE, it argues, offers web searchers a way to seek information in a conversational mode, but
ultimately keeps users in Google’s “walled garden” as it “plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as
Google’s general search service, GoogleBot.\n\nPlus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,”
citing both a 2023 report from the News Media Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with
researchers at the Allen Institute for AI, had found that News and Media sites were the third largest category of AI training data.)\n\nThe case points to other
concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat messages — an issue
raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.', 'News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nThe case also arrives alongside the filing of the U.S. Justice
Department’s lawsuit against Google for monopolizing digital ad technologies, and references the 2020 Justice Department’s civil antitrust suit over search and
search advertising (which are different markets from digital ad technologies in the more recent suit).\n\n“The anticompetitive effects of Google’s scheme cause
profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an announcement posted to the website of the law firm handling the
case, Hausfeld.\n\n“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore
and ensure competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial
intelligence,” it states.\n\nA Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.']
TechCrunch_related_entities: [{'entity': 'Alphabet', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle', 'relationship': 'MENTIONS'}, {'entity':
'U.S. District Court in D.C.', 'relationship': 'MENTIONS'}, {'entity': 'Search Generative Experience', 'relationship': 'MENTIONS'}, {'entity': 'Bard AI
chatbot', 'relationship': 'MENTIONS'}, {'entity': 'Knowledge Graph', 'relationship': 'MENTIONS'}, {'entity': 'anticompetitive behavior', 'relationship':
'MENTIONS'}, {'entity': 'U.S. antitrust law', 'relationship': 'MENTIONS'}, {'entity': 'Sherman Act', 'relationship': 'MENTIONS'}, {'entity': 'Arkansas',
'relationship': 'MENTIONS'}, {'entity': 'question-and-answer technologies', 'relationship': 'MENTIONS'}, {'entity': 'Publishers', 'relationship': 'MENTIONS'},
{'entity': 'Knowledge Panel', 'relationship': 'MENTIONS'}, {'entity': 'Featured Snippets', 'relationship': 'MENTIONS'}, {'entity': 'class action antitrust
suit', 'relationship': 'MENTIONS'}, {'entity': 'misappropriation of Publishers’ content', 'relationship': 'MENTIONS'}, {'entity': 'The Wall Street Journal',
'relationship': 'MENTIONS'}, {'entity': 'report on Thursday', 'relationship': 'MENTIONS'}, {'entity': 'deal to license news for AI model training',
'relationship': 'MENTIONS'}, {'entity': 'SGE', 'relationship': 'MENTIONS'}, {'entity': 'Google Bard AI', 'relationship': 'MENTIONS'}, {'entity': 'GoogleBot',
'relationship': 'MENTIONS'}, {'entity': 'Google general search service', 'relationship': 'MENTIONS'}, {'entity': 'AdSense', 'relationship': 'MENTIONS'},
{'entity': 'News Media Alliance', 'relationship': 'MENTIONS'}, {'entity': 'Washington Post', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games lawsuit against
Google over app store antitrust issues', 'relationship': 'MENTIONS'}, {'entity': 'app store antitrust issues', 'relationship': 'MENTIONS'}, {'entity': 'U.S.
Justice Department', 'relationship': 'MENTIONS'}, {'entity': 'Hausfeld', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle, LLC', 'relationship':
'MENTIONS'}, {'entity': 'class action antitrust suit against Google', 'relationship': 'MENTIONS'}, {'entity': 'U.S. Justice Department lawsuit against Google',
'relationship': 'MENTIONS'}, {'entity': '2020 Justice Department civil antitrust suit over search and search advertising', 'relationship': 'MENTIONS'},
{'entity': 'digital ad technologies', 'relationship': 'MENTIONS'}, {'entity': 'Clayton Act', 'relationship': 'MENTIONS'}, {'entity': 'search and search
advertising', 'relationship': 'MENTIONS'}, {'entity': 'digital news and reference publishing', 'relationship': 'MENTIONS'}, {'entity': 'class-wide monetary and
injunctive relief', 'relationship': 'MENTIONS'}, {'entity': 'free marketplace of ideas', 'relationship': 'MENTIONS'}, {'entity': 'free press', 'relationship':
'MENTIONS'}, {'entity': 'Epic Games', 'relationship': 'MENTIONS'}, {'entity': 'antitrust', 'relationship': 'MENTIONS'}, {'entity': 'The Atlantic',
'relationship': 'MENTIONS'}, {'entity': 'Axel Springer', 'relationship': 'MENTIONS'}, {'entity': 'Allen Institute for AI', 'relationship': 'MENTIONS'},
{'entity': 'class action lawsuit', 'relationship': 'MENTIONS'}, {'entity': 'consumers', 'relationship': 'MENTIONS'}, {'entity': 'AI', 'relationship':
'MENTIONS'}, {'entity': 'AI technologies', 'relationship': 'MENTIONS'}, {'entity': 'Similarweb', 'relationship': 'MENTIONS'}, {'entity': 'Google',
'relationship': 'MENTIONS'}, {'entity': 'artificial intelligence', 'relationship': 'MENTIONS'}, {'entity': 'D.C.', 'relationship': 'MENTIONS'}, {'entity':
'OpenAI', 'relationship': 'MENTIONS'}, {'entity': 'labor', 'relationship': 'MENTIONS'}]
publishers: ['TechCrunch']
Epic_Games_report: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today
Epic_Games_passages: ['What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nTo
make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.\n\nGoogle argues its commissions aren’t
just tied to billing, but offers discount for third-party billing options\n\nToday, Google requires apps to use its own first-party billing system, and charges
app developers a 15% to 30% commission on the sales it processes.\n\nHowever, Google will counter Epic’s arguments by pointing out that it has rolled out a new
option for app developers called User Choice Billing, which Epic has declined to use.', 'What to know about Fortnite maker Epic Games’ antitrust battle with
Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nThis program, still in pilot testing, is open to all developers who sell apps in the 35 markets
where it’s now available, including the U.S., and reduces the standard commission by 4% for companies who choose to use their own payment processing solution.
Spotify and Bumble were the initial testers for the new system, first introduced in November 2022, and Match, as part of its settlement agreement, will also now
take advantage of this option.\n\nGoogle will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other
features and controls to keep Android users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to
build and grow their apps. Google also helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging
from tablets to TVs to autos and more. And it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.', 'What to
know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nThe competitive landscape with other
app stores, OEMs\n\nThe tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point
out that not only does the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also
supports competitive app stores like Samsung’s Galaxy Store and Amazon’s Appstore.\n\nEpic, however, will attempt to demonstrate that Google makes it hard for
alternative app stores to succeed and reach consumers, noting that 90% of apps are still downloaded through Google Play. And it will point out that Google
bundles the Play Store with other Google apps that Android OEMs (original equipment manufacturers) have to agree to in order to use Android.\n\nEpic will also
bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps outside of Google
Play.', 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nEpic Games will
specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app. But OnePlus had
to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between Google and
Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed by
Samsung.\n\nEpic will argue Google’s developer agreements are anti-competitive\n\nA second set of agreements Epic will highlight are those between Google and
Android app developers themselves. One such program, the “Project Hug” initiative, a part of the Google Games Velocity Program, will be cited as an example of
where Google paid competitors not to distribute their apps on Android outside of the Play Store by targeting those that were most at risk of attrition from
Play, then approaching them with a monetary offer.', 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch,
2023-11-06T16:47:11+00:00)\nAs an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one
developer that wanted to go the route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020,
worth approximately $360 million, to keep the developer on Google Play, Epic will argue.\n\nGoogle, however, will characterize this program as a way to
incentivize developers to launch their apps on Android at the same time as they launch on iOS and plans to dispute Epic’s claims that it prevented developers
from opening their own app stores.\n\nAndroid allows sideloading\n\nGoogle will make note that, unlike Apple, it also allows apps to be sideloaded on Android
devices — something Epic takes advantage of today. It will note this process is streamlined and helps keep users safe.', 'What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nIt will note this process is streamlined and helps keep users safe.
But Epic will claim that this process is made to be overly difficult for end users, with multiple steps involved and “scare screens” that warn users of the
security risks involved with sideloading.\n\nEpic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the
record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.\n\nThe tech giant
will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5 billion users
without having to pay.\n\n“Android’s choice and flexibility work well for consumers and developers of all sizes.', 'What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\n“Android’s choice and flexibility work well for consumers and
developers of all sizes. We look forward to making our case in court as we fight to keep our users safe from harm, partner with developers to grow their
businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by Wilson White, VP, Government Affairs & Public
Policy.\n\nEpic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with
Match last week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will
include testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other
Android developers.\n\nGoogle is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.', 'What to know
about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nEpic, meanwhile, lost its antitrust
battle with Apple and is now asking the Supreme Court to weigh in.']
Epic_Games_related_entities: [{'entity': 'Sundar Pichai', 'relationship': 'MENTIONS'}, {'entity': 'U.S. antitrust law', 'relationship': 'MENTIONS'}, {'entity':
'Sherman Act', 'relationship': 'MENTIONS'}, {'entity': 'User Choice Billing', 'relationship': 'MENTIONS'}, {'entity': 'Google first-party billing system',
'relationship': 'MENTIONS'}, {'entity': 'Epic Games’ antitrust battle with Google', 'relationship': 'MENTIONS'}, {'entity': 'California’s anticompetitive laws',
'relationship': 'MENTIONS'}, {'entity': 'Cartwright Act', 'relationship': 'MENTIONS'}, {'entity': 'Unfair Competition Law', 'relationship': 'MENTIONS'},
{'entity': 'distribution of apps to Android users', 'relationship': 'MENTIONS'}, {'entity': 'payment processing solutions for content inside Android apps',
'relationship': 'MENTIONS'}, {'entity': 'Google Play', 'relationship': 'MENTIONS'}, {'entity': 'App Store', 'relationship': 'MENTIONS'}, {'entity': 'Android',
'relationship': 'MENTIONS'}, {'entity': 'Galaxy Store', 'relationship': 'MENTIONS'}, {'entity': 'Amazon Appstore', 'relationship': 'MENTIONS'}, {'entity':
'Android OEMs', 'relationship': 'MENTIONS'}, {'entity': 'Google apps', 'relationship': 'MENTIONS'}, {'entity': 'Anti-Fragmentation Agreements', 'relationship':
'MENTIONS'}, {'entity': 'Ninth Circuit', 'relationship': 'MENTIONS'}, {'entity': 'Ninth Circuit ruling in the Apple case', 'relationship': 'MENTIONS'},
{'entity': 'App store competition', 'relationship': 'MENTIONS'}, {'entity': 'OnePlus', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games app', 'relationship':
'MENTIONS'}, {'entity': 'Google Play Store', 'relationship': 'MENTIONS'}, {'entity': 'Android app developers', 'relationship': 'MENTIONS'}, {'entity': 'Project
Hug', 'relationship': 'MENTIONS'}, {'entity': 'Google Games Velocity Program', 'relationship': 'MENTIONS'}, {'entity': 'antitrust battle', 'relationship':
'MENTIONS'}, {'entity': 'agreement with OnePlus', 'relationship': 'MENTIONS'}, {'entity': 'deal between Google and Samsung', 'relationship': 'MENTIONS'},
{'entity': 'Google developer agreements', 'relationship': 'MENTIONS'}, {'entity': 'anti-competitive practices', 'relationship': 'MENTIONS'}, {'entity':
'Activision Blizzard', 'relationship': 'MENTIONS'}, {'entity': 'King.com', 'relationship': 'MENTIONS'}, {'entity': 'Candy Crush', 'relationship': 'MENTIONS'},
{'entity': 'three-year agreement', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games', 'relationship': 'MENTIONS'}, {'entity': 'Sideloading', 'relationship':
'MENTIONS'}, {'entity': 'Security risks', 'relationship': 'MENTIONS'}, {'entity': 'Off-the-record chats', 'relationship': 'MENTIONS'}, {'entity': 'Chat logs and
documents provided to the court', 'relationship': 'MENTIONS'}, {'entity': 'Play Store access and payment', 'relationship': 'MENTIONS'}, {'entity': 'Wilson
White', 'relationship': 'MENTIONS'}, {'entity': 'Match', 'relationship': 'MENTIONS'}, {'entity': 'U.S. attorneys general', 'relationship': 'MENTIONS'},
{'entity': "Epic's antitrust lawsuit", 'relationship': 'MENTIONS'}, {'entity': 'U.S. District Court in California’s Northern District', 'relationship':
'MENTIONS'}, {'entity': 'Department of Justice', 'relationship': 'MENTIONS'}, {'entity': "Google's antitrust lawsuit with the Department of Justice",
'relationship': 'MENTIONS'}, {'entity': 'search monopoly', 'relationship': 'MENTIONS'}, {'entity': 'antitrust', 'relationship': 'MENTIONS'}, {'entity':
'antitrust battle with Google', 'relationship': 'MENTIONS'}, {'entity': 'antitrust battle with Apple', 'relationship': 'MENTIONS'}, {'entity': 'Call of Duty',
'relationship': 'MENTIONS'}, {'entity': 'Netflix', 'relationship': 'MENTIONS'}, {'entity': 'Apple', 'relationship': 'MENTIONS'}, {'entity': 'Supreme Court',
'relationship': 'MENTIONS'}, {'entity': 'Bumble', 'relationship': 'MENTIONS'}, {'entity': 'Google Play billing', 'relationship': 'MENTIONS'}, {'entity': 'Tim
Sweeney', 'relationship': 'MENTIONS'}, {'entity': 'Samsung', 'relationship': 'MENTIONS'}, {'entity': 'Google', 'relationship': 'MENTIONS'}, {'entity': 'iOS',
'relationship': 'MENTIONS'}, {'entity': 'competition', 'relationship': 'MENTIONS'}, {'entity': 'Play Store', 'relationship': 'MENTIONS'}, {'entity': 'Spotify',
'relationship': 'MENTIONS'}, {'entity': 'U.S.', 'relationship': 'MENTIONS'}, {'entity': 'Amazon', 'relationship': 'MENTIONS'}, {'entity': 'Fortnite',
'relationship': 'MENTIONS'}]
shared_Google_entity: Google
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> U.S. antitrust law
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Google first-party billing system

Entities: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, antitrust, antitrust battle with Google,
antitrust battle with Apple, Apple, Supreme Court, TechCrunch, Google, Fortnite
</item 6>

<item 7>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.

Graph facts:
- Alphabet -ACCUSED_OF-> Sherman Act
- Alphabet -ACCUSED_OF-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Sherman Act
- Epic v. Google, explained -MENTIONS-> User Choice Billing

Entities: U.S. antitrust law, Sherman Act, User Choice Billing, Google first-party billing system, Epic Games’ antitrust battle with Google, California’s
anticompetitive laws, Cartwright Act, Unfair Competition Law, distribution of apps to Android users, payment processing solutions for content inside Android
apps, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, TechCrunch, Google
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Tencent
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Tencent
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Tencent
- European neobroker Scalable Capital raises $65M on a flat $1.4B valuation -MENTIONS-> Tencent

Entities: User Choice Billing, Tencent, Epic, China, Google, Epic v. Google, explained, Epic v. Google lawsuit, Judge Donato, Samsung Galaxy phone, The Verge
</item 8>

<item 9>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Epic Games
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Epic Games
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic Games
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Epic Games
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Epic Games

Entities: Epic Games, Sean, Apple, Google, Epic v. Google trial, Epic v. Apple trial, #FreeFortnite, Llamacorn, antitrust claims, illegal monopoly, Epic v.
Google, explained, Fortnite, The Verge
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Epic Games’ antitrust battle with Google, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic
Games, Sideloading, Security risks, Off-the-record chats, Chat logs and documents provided to the court, Play Store access and payment, TechCrunch, Google,
competition, Play Store, Fortnite
</item 10>
=== END ITEM id=af9b5840b8152dd0 ===
