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

=== ITEM id=4aa66c55be0f1479 ===
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
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> monopoly power
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Bing

Entities: monopoly power, FTC-Facebook case, Bing, Apple, WhatsApp, TechCrunch, data, Google antitrust case, 5 things we learned so far about the Google
antitrust case, DOJ, FTC, FTC lawsuit against Amazon, anticompetitive and unfair strategies, DOJ investigation of Apple, Apple's policy for third-party apps on
its devices
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- Amit Mehta -SAID-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Amit Mehta, APIs, third-party developers, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google
trial, Bing, TechCrunch, U.S. government, DOJ, antitrust law, Google, search, advertising
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> FTC-Facebook case
- Amit Mehta -PARTICIPATED_IN-> US v Google trial

Entities: Amazon antitrust lawsuit, FTC-Facebook case, Google antitrust lawsuit with Match Group, Google trial with Epic Games, Anticompetitive and unfair
strategies to maintain monopoly power, Vertical search, Pre-installs, Third-party apps, Commission structure, More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Firefox, anticompetitive behavior, Google Play
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Department of Justice, antitrust, Apple, online search, TechCrunch, New York, California, U.S. government, digital advertising, U.S.
government’s antitrust trial against Google, 2020 antitrust claim, January filing, eight states, 5 things we learned so far about the Google antitrust case
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
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bellshaw
- Microsoft -MENTIONS-> cloud gaming
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft

Entities: Bellshaw, Microsoft, TechCrunch, Pichai, Google antitrust case, message exchange, litigation hold, personnel matter, Internet Explorer, David
Drummond, Internet Explorer became the default search engine, Drummond’s letter, default settings, 5 things we learned so far about the Google antitrust case,
Google
</item 6>

<item 7>
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

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- Alphabet -ACCUSED_OF-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.

Entities: Sherman Act, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, U.S. Justice Department,
Hausfeld, Helena World Chronicle, LLC, class action antitrust suit against Google, U.S. Justice Department lawsuit against Google, 2020 Justice Department civil
antitrust suit over search and search advertising, digital ad technologies, Clayton Act, search and search advertising, digital news and reference publishing,
class-wide monetary and injunctive relief, free marketplace of ideas, free press
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.

Graph facts:
- Grey -WORKS_FOR-> DuckDuckGo
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> DuckDuckGo
- Is Google Search better than the rest? And is that fair? -MENTIONS-> DuckDuckGo
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> antitrust

Entities: third-party developers, DuckDuckGo, antitrust, Bing, Microsoft, Apple, monopoly, TechCrunch, data, Google antitrust case, Brave, Microsoft trial,
anticompetitive conduct, APIs, computer manufacturers
</item 8>

<item 9>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, TechCrunch, Pichai, Google antitrust case, Pichai's testimony, lucrative search agreement, systematically destroying chats, 5 things we learned
so far about the Google antitrust case, DOJ, Google, CNBC, Cook
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=4aa66c55be0f1479 ===

=== ITEM id=b75a84845492768a ===
Gold sentences and facts:
Sentence 0: Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right things.
  Fact 0.0: Erik ten Hag is a top coach from Ajax.
  Fact 0.1: The club needs a top blueprint not only for now but for the next five years.
  Fact 0.2: The club should work towards that blueprint.
  Fact 0.3: Sometimes on that journey there are hard moments when the club does not get the right results.
  Fact 0.4: The club can be doing the right things even when it does not get the right results.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United put in a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
"We are a long time from the Sir Alex Ferguson era. Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five
years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right
things.

"You have to know that, over that period, the consistency of the decision-making will yield a good end result. I think we've seen that both Manchester City and
Liverpool, that when you adopt that long-term strategy you can get real consistency and sustainable success.

"I think that's the biggest thing for Manchester United; putting down their identity of what they want to be today, but most crucially what they want to be in
five years' time, and where they want to be, and what they want to look like; from everything from young players, to the style of play, to the whole culture
peace in Manchester United."
</item 1>

<item 2>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Alex Ferguson
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Eric Cantona
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> riot police
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Turkey

Entities: Manchester United face Galatasaray with high hopes but bad memories, Galatasaray, Kobbie Mainoo, win against Everton, Alejandro Garnacho, Manchester
United, The Guardian, Everton, football
</item 2>

<item 3>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
They still have the FA Cup, and the extended pursuit of fourth or even fifth place, although the state of the coefficient means the thing that might keep
Manchester United out of Europe next season could be Manchester United being so bad in Europe this season. Maybe City can revive their fortunes by winning the
Champions League again.

Ten Hag will take the blame of course for the early exit and plenty of mistakes have been made, not least in the players the manager has been allowed to sign.
In reality assessing the job any United manager does with this team involves applying a series of filters. To be United manager is to undertake three concurrent
roles.

First you must manage the past, which is constantly in the room and which skews every act, achievement and expectation. Second you must manage the dysfunction
of the present, which expresses itself in every detail from a leaking stadium roof to the deep ills of a playing squad peopled with squatters and long‑term
missteps.

Graph facts:
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Champions League
- Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United -MENTIONS-> Champions League
- Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage -MENTIONS-> Champions League
- Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League

Entities: Champions League, Manchester United, FA Cup, The Guardian, City, Barney Ronay, Ten Hag’s Manchester United stand up and die at club stuck in the past
tense | Barney Ronay, Europe, Ten Hag
</item 3>

<item 4>
Graph query row:
article: Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United
published_on: 2023-10-16T21:00:00+00:00
connections: [{'relationship': 'PUBLISHED_BY', 'entity': 'talkSPORT'}, {'relationship': 'MENTIONS', 'entity': 'Champions League'}, {'relationship': 'MENTIONS',
'entity': 'Premier League'}, {'relationship': 'MENTIONS', 'entity': 'Southampton'}, {'relationship': 'MENTIONS', 'entity': 'Chelsea'}, {'relationship':
'MENTIONS', 'entity': 'Manchester United'}, {'relationship': 'MENTIONS', 'entity': 'England'}, {'relationship': 'MENTIONS', 'entity': 'Manchester'},
{'relationship': 'MENTIONS', 'entity': 'Daniel Levy'}, {'relationship': 'MENTIONS', 'entity': 'Jim Ratcliffe'}]
passages: [{'passage': 'Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United
(TalkSport, 2023-10-16T21:00:00+00:00)\nJim Ratcliffe’s move to Monaco in 2020 may have been bad news for the British taxpayer, but not for Manchester United
fans.\n\nThe chemical billionaire looks set to buy a 25 per cent stake in the Premier League giants after fellow bidder, Sheikh Jassim, withdrew his offer.\n\n6
Ratcliffe could still have a big say at United Credit: getty\n\n6 His wisest move could be to land his old neighbour Mitchell\n\nThat news hasn’t been too
warmly welcomed by United supporters, who would still have to put up with the hugely unpopular Glazer family in control of the club.\n\nIt could, though, mean
some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.\n\nShould Ratcliffe come in, talkSPORT
understands that he would take charge of footballing matters, something the Glazers have been accused of having little to no interest in.', 'page': 'Paul
Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)'}, {'passage': 'Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for
Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\nAlready a top tier European owner of OGC Nice, there has been some success, but nothing compared to
their French riviera neighbours, where Ratcliffe appears set to look for inspiration.\n\nPaul Mitchell recently left the principality of Monaco, but he could
well be back there in meetings with Ratcliffe who is planning to appoint a sporting director.\n\nMitchell fulfilled that role at Monaco after previous stints at
Leipzig, Tottenham, Southampton and MK Dons, becoming one of football’s most respected executives.\n\nA former player in England’s lower leagues, Mitchell was
previously linked with Chelsea and Liverpool, and much like Ratcliffe, he’s also from Greater Manchester.\n\nStarring mainly for Wigan and MK Dons, it was with
the latter where Mitchell was moved upstairs, and made his first big move by spotting future England regular Dele Alli after being named their head of
recruitment in 2010.', 'page': 'Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United
(TalkSport, 2023-10-16T21:00:00+00:00)'}, {'passage': "Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan
for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\nTwo years later he joined Southampton in the same role, and that’s where he really began to show
off, landing the Saints a number of players who would not only go on to become Champions League regulars, but make the club a fortune and ensure a Premier
League best finish.\n\nShowcasing his skills in casting his scouting net far and wide, Sadio Mane, Dusan Tadic, Graziano Pelle, Dejan Lovren and Victor Wanyama
all went on to secure Southampton a bright future, and Mitchell a promotion to Tottenham in 2014.\n\n6 Mane was one of Mitchell's early successes\n\n6 But
there's been many more Credit: Getty\n\nThe north London side had just unsuccessfully splurged their world record £85million from Real Madrid for Gareth Bale,
and it looked like Mitchell couldn’t have picked a worse time.", 'page': 'Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and
has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)'}, {'passage': 'Paul Mitchell signed Tottenham and Chelsea superstars, fell
out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\nOnly leaving the side to reach the pinnacle of his
job, finally stepping up from a head of recruitment to become sporting director at Monaco, Mitchell showed just how big picture he is with an interview in
2021.\n\n“I want to continue to drive standards in each and every department,” he said. “Psychology, sports science, nutrition, medicine, the way we travel, the
way we refuel.\n\n“Everything, every element I think we need to keep pushing to its maximum to try and be the best we possibly can.”\n\nHaving worked in
football management for over a decade, Mitchell’s career has always been about the long term for his club’s, with the young signings he’s made often only paying
off after he’s left.\n\nOne in particular was 2022/23 Premier League top scorer Son, who perfectly sums up his philosophy.\n\n“I took quite a lot of criticism
in my first year at Tottenham for some of the decisions and Sonny was kind of at the centre of that,” he explained.', 'page': 'Paul Mitchell signed Tottenham
and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)'}, {'passage': 'Paul
Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)\n“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen
but there was this narrative that started to grow internally and externally that he wasn’t a good finisher.\n\n6 Mitchell is becoming an increasingly
recognisable face\n\n“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both
sides’.”\n\nMitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team
game while the German was interim coach at Man United.\n\nSo astute is Mitchell in his observations, he even laid out a blueprint for United\'s success while at
Monaco.\n\n“I think context is important to measure what Manchester United\'s current situation is, but I think they do need to get to the point of really
putting down what Manchester United needs to be in the modern game,” he explained.\n\n"We are a long time from the Sir Alex Ferguson era.', 'page': 'Paul
Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)'}]
</item 4>

<item 5>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?

Graph facts:
- VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see -MENTIONS-> Spurs
- Gregg Popovich -WORKS_FOR-> Spurs
- Spurs -PARTICIPATED_IN-> Spurs vs Brighton

Entities: Spurs, Manchester United, West Ham, Ange-ball, Peter Schmeichel, Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays
off again as Spurs sink Everton, Optus Sport, Everton, The Roar | Sports Writers Blog, Ten Hag
</item 5>

<item 6>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey

Entities: From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Greig Cowie, Manchester City, Manchester United, Google, Ratcliffe,
The Guardian, the Glazers, Liverpool, Shutterstock, Football Daily, Google reCaptcha, Ten Hag, communications
</item 6>

<item 7>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
In 2020‑21, they lost their final two group games and were eliminated, while back in 2015-16, Wolfsburg won the deciding group match to knock out United.

Mauro Icardi celebrates scoring the winner for Galatasaray in their 3-2 win at Old Trafford in early October. Photograph: Darren Staples/AFP/Getty Images

When it comes to pressure in the Champions League of late, United do not cope well and there are few events trickier than a night in Istanbul. Pallister smiled
at a bell boy at the team hotel and got a finger across the throat in response, Brian McClair had a cleaner bang on his bedroom ceiling in the night and other
players received numerous phone calls to disrupt their sleep. The current squad could face the same level of malevolence and need to be prepared.

United have done little learning in recent times, repeating the same failures in Europe’s elite competition.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Alex Ferguson
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Eric Cantona
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> riot police
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Turkey

Entities: Manchester United face Galatasaray with high hopes but bad memories, Galatasaray, Wolfsburg, Mauro Icardi, Pallister, Brian McClair, Darren Staples,
Galatasaray's 3-2 win at Old Trafford, Manchester United's final two group games, 2015-16 deciding group match, Champions League, AFP, Manchester United, Old
Trafford, The Guardian
</item 7>

<item 8>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative
that started to grow internally and externally that he wasn’t a good finisher.

6 Mitchell is becoming an increasingly recognisable face

“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”

Mitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team game while
the German was interim coach at Man United.

So astute is Mitchell in his observations, he even laid out a blueprint for United's success while at Monaco.

“I think context is important to measure what Manchester United's current situation is, but I think they do need to get to the point of really putting down what
Manchester United needs to be in the modern game,” he explained.

"We are a long time from the Sir Alex Ferguson era.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Manchester United, Daniel Levy, Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for
Manchester United, Paul Mitchell, Monaco, Leipzig, Sonny, Leverkusen, Rangnick, youth team game, talkSPORT, Sir Alex Ferguson, Hamburg, Tottenham
</item 8>

<item 9>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"So I think there's a lot of question marks in the performances and the way they're going about things at the moment but balance for me is one of the key
aspects that they've got to really find.

"We mentioned in commentary, scars from throwing away leads, those are hard to get away, you've got to really manage games and he mentioned that himself Bruno,
you've got to go out managing games correctly.

"They've got to do that not over one or two or three games, they've got to do that over a period of games now so they can erase those scars that they've created
and they'e grown through bad results and throwing away leads, especially in Europe."

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Hakim Ziyech
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Kerem
Akturkoglu
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> 3-3 draw with
Galatasaray
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> TNT Sports
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> final match

Entities: Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe, Bruno, Manchester
United, talkSPORT, Europe
</item 9>

<item 10>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Ziyech
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> defence
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> final match
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> TNT Sports
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Gabby
Agbonlahor

Entities: Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe, Hakim Ziyech,
Kerem Akturkoglu, 3-3 draw with Galatasaray, Galatasaray, Chelsea, Alejandro Garnacho, Manchester United, Erik ten Hag, Champions League, Scott McTominay,
talkSPORT, Bruno Fernandes, Europe, Andre Onana
</item 10>
=== END ITEM id=b75a84845492768a ===

=== ITEM id=16685513d74ba6f7 ===
Gold sentences and facts:
Sentence 0: Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
  Fact 0.0: Jim Harbaugh was suspended.
  Fact 0.1: Jim Harbaugh held his usual weekly news conference on Monday despite the suspension.
  Fact 0.2: Jim Harbaugh mostly avoided addressing the only topic on anyone's mind at the news conference.
Sentence 1: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 1.0: The suspension would be Harbaugh's second suspension of the season.
  Fact 1.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 1.2: The self-imposed ban occurred amid an NCAA investigation.
  Fact 1.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 2: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 2.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 2.1: The leading piece of drama now returns.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Conference has confirmed that it is not aware of any information suggesting Coach Harbaugh's involvement in the allegations. The University continues to
cooperate fully with the NCAA's investigation.

From the Big Ten:

"The Big Ten Conference's commitment to student-athletes, sportsmanship and the commissioner's duty to protect the integrity of competition will never waver.
Today's decision by the University of Michigan to withdraw its legal challenge against the conference's November 10th notice of disciplinary action is
indicative of the high standards and values that the conference and the university seek to uphold, the University of Michigan is a valued member of the Big Ten
Conference and the conference will continue to work cooperatively with the university and the NCAA during this process."

Monday, Nov. 13

Harbaugh says he'll state his case at Friday hearing

Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.

Graph facts:
- Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Jim Harbaugh
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Jim Harbaugh
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> sign-stealing scandal
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Jim Harbaugh
- Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game -MENTIONS-> Jim Harbaugh

Entities: sign-stealing scandal, Jim Harbaugh, Big Ten Conference, University of Michigan, University of Michigan legal challenge withdrawal, University of
Michigan notice of disciplinary action, Jim Harbaugh news conference, Jim Harbaugh hearing, cheating investigation, Michigan sign-stealing scandal news: Latest
updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, NCAA investigation, NCAA
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
12

U-M president praises Wolverines for handling 'challenges and adversity'

The day after the Penn State win, University of Michigan President Santa Ono posted on social media, praising the team's ability to "respond to any challenge
head on with a conviction to do better and to emerge even stronger."

Countless members of the University of Michigan family have reached out to me over the weekend and I wanted to express my appreciation. Like any community, we
face our share of challenges and adversity. There have been many such moments in our history. But as our team showed so… — Santa Ono (@SantaJOno) November 12,
2023

The post drew immediate and extensive blowback on social media and beyond, as observers questioned Ono's tone with Michigan in the midst of a self-inflicted
controversy.

Saturday, Nov. 11

Warde Manuel issues strong statement against Big Ten

Michigan athletic director Warde Manuel did not mince words in a statement on the Big Ten's suspension of Harbaugh.

Graph facts:
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Big Ten
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Big Ten
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Big Ten
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> sign-stealing scandal
- Can Alabama make College Football Playoff? How Texas, Oregon losses could impact Tide's 2023 chances -MENTIONS-> Big Ten

Entities: sign-stealing scandal, Big Ten, Santa Ono, Jim Harbaugh, University of Michigan, Wolverines, Penn State win, Big Ten suspension of Harbaugh, Warde
Manuel, cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation,
Sporting News, NCAA
</item 2>

<item 3>
Graph query row:
article: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
published_on: 2023-11-11T22:50:13+00:00
passage: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
The Big Ten announced Michigan coach Jim Harbaugh will be prohibited from coaching on the sideline for the remainder of the regular season.

At least, in theory. The Friday ruling is the next chapter in an NCAA investigation into Michigan for in-person scouting and sign stealing in a scandal that led
to the resignation of staffer Connor Stallions on Nov. 3. Harbaugh will be allowed to coach during the week but will not be able to be on the sideline. Michigan
did not get a ruling on a temporary restraining order ahead of the showdown against No. 10 Penn State.

The Wolverines still won 24-15 under interim coach Sherrone Moore, but the fallout from the scandal continues to have a far-reaching impact on the Big Ten,
Michigan and college football beyond this year's College Football Playoff race.

A look at the winners and losers from the scandal to date:

Michigan scandal winners and losers

Winners

Tony Petitti

Petitti certainly did not hit Michigan with a "slap on the wrist."
</item 3>

<item 4>
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

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Penn State, Big Ten, Happy Valley, Michigan sign-stealing scheme investigation, Jim Harbaugh suspension, Jim Harbaugh, Michigan
sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Connor Stalions, Sporting News, sign-stealing
scheme, NCAA
</item 4>

<item 5>
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
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Nicole Auerbach (@NicoleAuerbach) November 11, 2023

Harbaugh hearing coming down to wire before Penn State game

ESPN's Pete Thamel reported it's unclear Harbaugh will be granted a hearing in time to be able to coach against Penn State. With kickoff scheduled for noon ET,
Thamel said "the drama's going to go right up 'till noon."

Latest from ⁦@CollegeGameDay⁩ on Jim Harbaugh’s TRO legal filing. Will he coach Michigan at Penn State today? pic.twitter.com/bz3R1eYxI9 — Pete Thamel
(@PeteThamel) November 11, 2023

The case is reportedly in the hands of Judge Carol Kunhke, a Michigan alumnus.

Harbaugh was not on the Michigan team bus with other coaches and players when it arrived at Beaver Stadium Saturday morning.

Michigan players and coaches arrive. No Jim Harbaugh on the bus.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Nicole Auerbach, Penn State, Michigan sign-stealing scandal, Beaver Stadium, ESPN, Jim Harbaugh, NCAA cheating investigation, Carol Kunhke,
Harbaugh hearing, TRO legal filing, Penn State game, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating
investigation, Pete Thamel, Sporting News
</item 6>

<item 7>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Penn State, Big Ten, Jim Harbaugh suspension, Jim Harbaugh, NCAA cheating investigation, Michigan game vs. Penn State, temporary restraining
order filing, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Pete Thamel, hearing,
Sporting News, NCAA
</item 7>

<item 8>
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
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.



"Michigan has been found in violation of the Big Ten Sportsmanship Policy for conducting an impermissible, in-person scouting operation over multiple years..."
https://t.co/htnUkshmnF — Chris Vannini (@ChrisVannini) November 10, 2023

Jim Harbaugh joins Michigan for trip to State College

According to the sleuths on the ground in Ann Arbor, Harbaugh is on the team bus and heading to the airport with his team as Michigan awaits word on a potential
suspension.

Jim Harbaugh boarding the bus pic.twitter.com/CWBJbwEEhi — Austin Meek (@byAustinMeek) November 10, 2023

Michigan still awaiting word from Big Ten

According to ESPN's Adam Rittenberg, Michigan still hasn't heard about an impending punishment from the Big Ten as its 1 p.m. ET flight looms.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Big Ten, ESPN, Jim Harbaugh suspension, Jim Harbaugh, Chris Vannini, Big Ten Sportsmanship Policy, impermissible, in-person scouting
operation, Austin Meek, Ann Arbor, State College, Adam Rittenberg, NCAA cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible
Jim Harbaugh suspension from NCAA cheating investigation, Sporting News
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
ET flight looms.

Michigan's administration has not heard from the Big Ten as of this hour, per sources. The team will begin its preparations to leave for Penn State around 1
p.m. ET. Really going down to the wire with any discipline, which, as several sources noted, isn't really fair to players. — Adam Rittenberg (@ESPNRittenberg)
November 10, 2023

Harbaugh still expected to fly to State College

Dinich reported earlier on Friday that punishment to Michigan could come around the time the team is expected to get on a plane and fly to Pennsylvania for its
game against Penn State. But regardless of when the punishment is announced, Harbaugh's attorney, Tom Mars, reportedly told ESPN the coach will be headed to
State College.

"I'm confident he's going to be on that plane one way or another," Mars reportedly told ESPN.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Penn State, Big Ten, suspension, Jim Harbaugh, State College, game against Penn State, Tom Mars, Dinich, Adam
Rittenberg, cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation,
ESPN, Sporting News
</item 10>
=== END ITEM id=16685513d74ba6f7 ===
